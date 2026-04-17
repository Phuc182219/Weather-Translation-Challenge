"""
Weather Translation Challenge - Solution (v8)

Sequence-to-sequence translation of 72-hour weather observations between
weather stations, trained from scratch.

Context
-------
v5 (climate-anomaly BiLSTM ensemble) scored Shipd test 0.7281.
v7 (coupled pair-dropout ensemble) val 0.8411 diverse subset; the
partial-ensemble-after-7 peak was 0.8417. Per-seed LSTM val MSE
clustered at 0.1244-0.1256 -> near the capacity limit of v7's 2.5M-param
BiLSTM. To push past ~0.84 val (needed for top-2 rank) we have to raise
the per-seed ceiling, not the ensemble count.

v8 changes vs v7
----------------
1. *Bigger BiLSTM.* d_model 192 -> 224, station_dim 48 -> 64 (~3.3M
   params). Extra capacity for the anomaly translation.
2. *Physical features.* Relative humidity (August-Roche-Magnus) and
   vapor pressure deficit are computed from temp+dewpoint and fed in as
   extra per-hour channels. Both are deterministic functions of the raw
   inputs, but exponential/nonlinear, so explicit channels reduce the
   effective function the BiLSTM has to fit.
3. *Longer training.* 150 epochs (was 120) with the same warmup+cosine
   schedule — v7's loss curve was still slowly improving at epoch 120.
4. *Test-time augmentation.* 3 forward passes per test sample with small
   Gaussian noise (sigma=0.02 in anomaly-norm space), averaged.
5. *Trimmed ensemble.* 5 LSTM pd=0 + 1 GRU pd=0 + 1 LSTM pd=0.3 = 7
   models — matching the partial peak from v7 analysis.

Climate-anomaly framework from v5 preserved unchanged.
"""

import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "public")
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    for cand in (
        os.path.join(HERE, "dataset", "public"),
        "/kaggle/input/dataset/public",
        os.path.join(HERE, "..", "dataset", "public"),
    ):
        if os.path.exists(os.path.join(cand, "train.csv")):
            DATA_DIR = cand
            break

OUT_DIR = os.path.join(HERE, "working")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

VARS = ("temp", "dewpoint", "wind_speed")
T = 72
N_VARS = len(VARS)

# Input channels fed to the model:
#   0: temp (anomaly-normalised)
#   1: dewpoint (anomaly-normalised)
#   2: wind_speed (anomaly-normalised)
#   3: relative humidity (0..1 scaled)
#   4: vapor pressure deficit (normalised)
N_INPUT_CHANNELS = N_VARS + 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reshape_wide_to_series(df: pd.DataFrame, role: str) -> np.ndarray:
    arrs = []
    for v in VARS:
        cols = [f"{role}_{v}_{i}" for i in range(T)]
        arrs.append(df[cols].values.astype(np.float32))
    return np.stack(arrs, axis=-1)


def stations_list(train_df: pd.DataFrame, test_df: pd.DataFrame):
    s = set(train_df["source_city"]).union(train_df["target_city"])
    s = s.union(test_df["source_city"]).union(test_df["target_city"])
    return sorted(s)


def compute_station_climate(train_df: pd.DataFrame, stn_to_id: dict):
    n = len(stn_to_id)
    sums = np.zeros((n, N_VARS), dtype=np.float64)
    counts = np.zeros((n, N_VARS), dtype=np.float64)

    Xsrc = reshape_wide_to_series(train_df, "source")
    Xtgt = reshape_wide_to_series(train_df, "target")
    s_ids = np.array([stn_to_id[s] for s in train_df["source_city"]])
    t_ids = np.array([stn_to_id[s] for s in train_df["target_city"]])

    for i, sid in enumerate(s_ids):
        sums[sid] += Xsrc[i].sum(axis=0)
        counts[sid] += T
    for i, tid in enumerate(t_ids):
        sums[tid] += Xtgt[i].sum(axis=0)
        counts[tid] += T

    counts = np.clip(counts, 1, None)
    return (sums / counts).astype(np.float32)


def compute_anomaly_std(train_df: pd.DataFrame, stn_to_id: dict, climate: np.ndarray):
    Xsrc = reshape_wide_to_series(train_df, "source")
    Xtgt = reshape_wide_to_series(train_df, "target")
    s_ids = np.array([stn_to_id[s] for s in train_df["source_city"]])
    t_ids = np.array([stn_to_id[s] for s in train_df["target_city"]])
    anom_src = Xsrc - climate[s_ids][:, None, :]
    anom_tgt = Xtgt - climate[t_ids][:, None, :]
    combined = np.concatenate([anom_src, anom_tgt], axis=0).reshape(-1, N_VARS)
    return combined.std(axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Physical features (deterministic, derived from temp + dewpoint)
# ---------------------------------------------------------------------------
# August-Roche-Magnus saturation vapor pressure (hPa): e_s(T) = 6.1094 *
# exp(17.625 * T / (243.04 + T)). Relative humidity = e_s(dp) / e_s(T).
def saturation_vp(temp_c: np.ndarray) -> np.ndarray:
    return 6.1094 * np.exp(17.625 * temp_c / (243.04 + temp_c))


def relative_humidity(temp_c: np.ndarray, dp_c: np.ndarray) -> np.ndarray:
    es_t = saturation_vp(temp_c)
    es_d = saturation_vp(dp_c)
    rh = np.clip(es_d / np.clip(es_t, 1e-4, None), 0.0, 1.5)
    return rh.astype(np.float32)


def vapor_pressure_deficit(temp_c: np.ndarray, dp_c: np.ndarray) -> np.ndarray:
    # VPD = e_s(T) - e(dp); positive when air is drier than saturated.
    return (saturation_vp(temp_c) - saturation_vp(dp_c)).astype(np.float32)


def augment_with_physical(temp_raw: np.ndarray, dp_raw: np.ndarray,
                          ws_anom_n: np.ndarray, temp_anom_n: np.ndarray,
                          dp_anom_n: np.ndarray,
                          rh_mean: float, rh_std: float,
                          vpd_mean: float, vpd_std: float):
    """Given per-hour raw temp/dp (degrees C) and already-normalised
    anomaly channels, produce (N, T, N_INPUT_CHANNELS) feature array."""
    rh = relative_humidity(temp_raw, dp_raw)
    vpd = vapor_pressure_deficit(temp_raw, dp_raw)
    rh_n = (rh - rh_mean) / (rh_std + 1e-6)
    vpd_n = (vpd - vpd_mean) / (vpd_std + 1e-6)
    out = np.stack([temp_anom_n, dp_anom_n, ws_anom_n, rh_n, vpd_n], axis=-1)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AnomalyDataset(Dataset):
    def __init__(self, src_feats, tgt_anom_n, s_ids, t_ids):
        """src_feats: (N, T, N_INPUT_CHANNELS). Only the first N_VARS
        channels are residual-target candidates (we never add-back the
        RH/VPD channels — those are pure inputs). tgt_anom_n:
        (N, T, N_VARS) or None for test."""
        self.src = src_feats
        self.tgt = tgt_anom_n
        self.s_ids = s_ids
        self.t_ids = t_ids

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        sx = self.src[idx]
        ty = self.tgt[idx] if self.tgt is not None else np.zeros((T, N_VARS), dtype=np.float32)
        return (
            torch.from_numpy(sx),
            torch.from_numpy(ty),
            int(self.s_ids[idx]),
            int(self.t_ids[idx]),
        )


# ---------------------------------------------------------------------------
# Time features
# ---------------------------------------------------------------------------
def sinusoidal_time_feats(T: int) -> torch.Tensor:
    hours = torch.arange(T, dtype=torch.float32)
    return torch.stack(
        [
            torch.sin(2 * math.pi * hours / 24.0),
            torch.cos(2 * math.pi * hours / 24.0),
            torch.sin(2 * math.pi * hours / 72.0),
            torch.cos(2 * math.pi * hours / 72.0),
        ],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# Recurrent translator
# ---------------------------------------------------------------------------
class RecurrentTranslator(nn.Module):
    def __init__(
        self,
        n_stations,
        d_model=224,
        n_layers=3,
        dropout=0.2,
        station_dim=64,
        pair_dropout=0.0,
        rnn_type="lstm",
        n_input_channels=N_INPUT_CHANNELS,
    ):
        super().__init__()
        self.src_station_emb = nn.Embedding(n_stations, station_dim)
        self.tgt_station_emb = nn.Embedding(n_stations, station_dim)
        self.null_src_emb = nn.Parameter(torch.zeros(station_dim))
        self.null_tgt_emb = nn.Parameter(torch.zeros(station_dim))
        nn.init.trunc_normal_(self.null_src_emb, std=0.02)
        nn.init.trunc_normal_(self.null_tgt_emb, std=0.02)
        self.pair_dropout = pair_dropout
        self.rnn_type = rnn_type
        self.n_input_channels = n_input_channels

        in_feats = n_input_channels + 4 + 2 * station_dim
        self.input_proj = nn.Linear(in_feats, d_model)
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.rnn = rnn_cls(
            input_size=d_model, hidden_size=d_model, num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True, batch_first=True,
        )
        self.norm = nn.LayerNorm(2 * d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, N_VARS),
        )
        self.register_buffer("time_feats", sinusoidal_time_feats(T))

    def forward(self, x, s_ids, t_ids):
        B = x.size(0)
        s_e = self.src_station_emb(s_ids)
        t_e = self.tgt_station_emb(t_ids)

        if self.pair_dropout >= 1.0:
            s_e = self.null_src_emb.unsqueeze(0).expand(B, -1)
            t_e = self.null_tgt_emb.unsqueeze(0).expand(B, -1)
        elif self.training and self.pair_dropout > 0:
            mask = (torch.rand(B, device=x.device) < self.pair_dropout).float().unsqueeze(-1)
            s_e = s_e * (1 - mask) + self.null_src_emb.unsqueeze(0) * mask
            t_e = t_e * (1 - mask) + self.null_tgt_emb.unsqueeze(0) * mask

        station_feat = torch.cat([s_e, t_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        tf = self.time_feats.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, tf, station_feat], dim=-1)
        h = self.input_proj(tokens)
        h, _ = self.rnn(h)
        h = self.norm(h)
        return self.head(h)


# Back-compat alias
BiLSTMTranslator = RecurrentTranslator


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------
def train_one(
    seed,
    train_ds,
    val_ds,
    test_ds,
    n_stations,
    epochs=150,
    pair_dropout=0.0,
    rnn_type="lstm",
    tta_passes=3,
    tta_sigma=0.02,
):
    set_seed(seed)
    BATCH = 128
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = RecurrentTranslator(
        n_stations=n_stations, pair_dropout=pair_dropout, rnn_type=rnn_type,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{rnn_type} seed={seed} pair_dropout={pair_dropout}] "
          f"Model params: {n_params:,}", flush=True)

    LR = 2e-3
    WD = 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = epochs * steps_per_epoch
    warmup = steps_per_epoch * 3

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    best_val = float("inf")
    best_state = None
    step = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for sx, ty, s_id, t_id in train_loader:
            sx, ty = sx.to(DEVICE), ty.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
            for g in opt.param_groups:
                g["lr"] = LR * lr_at(step)

            delta = model(sx, s_id, t_id)
            # Residual prediction: add delta to the FIRST N_VARS channels of
            # the source (the anomaly-norm temp/dp/ws), not to the derived
            # physical channels.
            pred = sx[..., :N_VARS] + delta
            loss = F.mse_loss(pred, ty, reduction="mean")

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * sx.size(0)
            tr_n += sx.size(0)
            step += 1
        tr_loss /= max(1, tr_n)

        if val_loader is not None:
            model.eval()
            vl = 0.0
            vn = 0
            with torch.no_grad():
                for sx, ty, s_id, t_id in val_loader:
                    sx, ty = sx.to(DEVICE), ty.to(DEVICE)
                    s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
                    pred = sx[..., :N_VARS] + model(sx, s_id, t_id)
                    vl += F.mse_loss(pred, ty, reduction="mean").item() * sx.size(0)
                    vn += sx.size(0)
            vl /= max(1, vn)
            improved = vl < best_val - 1e-6
            flag = "*" if improved else " "
            if ep == 1 or ep % 10 == 0 or improved:
                print(
                    f"[{rnn_type} seed={seed} pd={pair_dropout}] Ep {ep:3d}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  "
                    f"train={tr_loss:.4f}  val={vl:.4f} {flag}  "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
            if improved:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(
                    f"[{rnn_type} seed={seed} pd={pair_dropout}] Ep {ep:3d}  "
                    f"lr={opt.param_groups[0]['lr']:.2e}  train={tr_loss:.4f}  "
                    f"elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    # Inference with TTA: average (tta_passes-1) noisy passes + 1 clean pass.
    def predict(loader):
        preds = []
        with torch.no_grad():
            for sx, _ty, s_id, t_id in loader:
                sx = sx.to(DEVICE)
                s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
                # Clean pass
                accum = sx[..., :N_VARS] + model(sx, s_id, t_id)
                count = 1
                for _ in range(tta_passes - 1):
                    # Add small Gaussian noise to ONLY the 3 anomaly channels
                    noise = torch.randn_like(sx[..., :N_VARS]) * tta_sigma
                    sx_aug = sx.clone()
                    sx_aug[..., :N_VARS] = sx[..., :N_VARS] + noise
                    accum = accum + sx_aug[..., :N_VARS] + model(sx_aug, s_id, t_id)
                    count += 1
                preds.append((accum / count).cpu().numpy())
        return np.concatenate(preds, axis=0)

    preds_te = predict(test_loader)
    preds_val = predict(val_loader) if val_loader is not None else None
    return preds_te, preds_val, best_val


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------
def write_submission(test_df, preds, out_path):
    cols = ["id"]
    out = {"id": test_df["id"].values}
    for i in range(T):
        for vi, v in enumerate(VARS):
            cols.append(f"target_{v}_{i}")
            out[f"target_{v}_{i}"] = preds[:, i, vi]
    sub = pd.DataFrame(out, columns=cols)
    sub.to_csv(out_path, index=False)
    print(f"Wrote submission: {out_path}  shape={sub.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")

    stations = stations_list(train_df, test_df)
    stn_to_id = {s: i for i, s in enumerate(stations)}
    n_stations = len(stations)
    print(f"Stations: {stations}")

    climate = compute_station_climate(train_df, stn_to_id)
    anom_std = compute_anomaly_std(train_df, stn_to_id, climate)
    print(f"Anomaly std per var: {anom_std}")

    Xsrc = reshape_wide_to_series(train_df, "source")
    Ytgt = reshape_wide_to_series(train_df, "target")
    s_ids = np.array([stn_to_id[s] for s in train_df["source_city"]], dtype=np.int64)
    t_ids = np.array([stn_to_id[s] for s in train_df["target_city"]], dtype=np.int64)

    src_anom_n = ((Xsrc - climate[s_ids][:, None, :]) / (anom_std + 1e-6)).astype(np.float32)
    tgt_anom_n = ((Ytgt - climate[t_ids][:, None, :]) / (anom_std + 1e-6)).astype(np.float32)

    Xte = reshape_wide_to_series(test_df, "source")
    s_ids_te = np.array([stn_to_id[s] for s in test_df["source_city"]], dtype=np.int64)
    t_ids_te = np.array([stn_to_id[s] for s in test_df["target_city"]], dtype=np.int64)
    src_anom_te_n = ((Xte - climate[s_ids_te][:, None, :]) / (anom_std + 1e-6)).astype(np.float32)

    # Physical features — computed from RAW temp/dewpoint, then normalised
    # with stats from the full training source+target pool so they are
    # comparable between train and test.
    def compute_rh_vpd(T_raw, dp_raw):
        return relative_humidity(T_raw, dp_raw), vapor_pressure_deficit(T_raw, dp_raw)

    rh_src_tr, vpd_src_tr = compute_rh_vpd(Xsrc[..., 0], Xsrc[..., 1])
    rh_tgt_tr, vpd_tgt_tr = compute_rh_vpd(Ytgt[..., 0], Ytgt[..., 1])
    rh_all = np.concatenate([rh_src_tr.ravel(), rh_tgt_tr.ravel()])
    vpd_all = np.concatenate([vpd_src_tr.ravel(), vpd_tgt_tr.ravel()])
    rh_mean, rh_std = float(rh_all.mean()), float(rh_all.std() + 1e-6)
    vpd_mean, vpd_std = float(vpd_all.mean()), float(vpd_all.std() + 1e-6)
    print(f"RH  mean={rh_mean:.3f} std={rh_std:.3f}")
    print(f"VPD mean={vpd_mean:.3f} std={vpd_std:.3f}")

    # Build (N, T, N_INPUT_CHANNELS) feature arrays for train, val, test.
    src_feats_tr = augment_with_physical(
        Xsrc[..., 0], Xsrc[..., 1],
        src_anom_n[..., 2], src_anom_n[..., 0], src_anom_n[..., 1],
        rh_mean, rh_std, vpd_mean, vpd_std,
    )
    src_feats_te = augment_with_physical(
        Xte[..., 0], Xte[..., 1],
        src_anom_te_n[..., 2], src_anom_te_n[..., 0], src_anom_te_n[..., 1],
        rh_mean, rh_std, vpd_mean, vpd_std,
    )
    print(f"Input features per hour: {N_INPUT_CHANNELS} "
          f"(temp, dewpoint, wind_speed anom-norm + RH, VPD)")

    SEED0 = 1337
    set_seed(SEED0)
    rng = np.random.default_rng(SEED0)
    pair_keys = (train_df["source_city"] + "->" + train_df["target_city"]).values
    val_mask = np.zeros(len(train_df), dtype=bool)
    for p in np.unique(pair_keys):
        idx = np.where(pair_keys == p)[0]
        take = max(1, int(0.1 * len(idx)))
        chosen = rng.choice(idx, size=take, replace=False)
        val_mask[chosen] = True

    train_ds = AnomalyDataset(src_feats_tr[~val_mask], tgt_anom_n[~val_mask],
                               s_ids[~val_mask], t_ids[~val_mask])
    val_ds = AnomalyDataset(src_feats_tr[val_mask], tgt_anom_n[val_mask],
                             s_ids[val_mask], t_ids[val_mask])
    test_ds = AnomalyDataset(src_feats_te, None, s_ids_te, t_ids_te)
    print(f"Train/Val/Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    ref_var = float(Ytgt.var())
    print(f"Ref var (proxy): {ref_var:.3f}")

    # 7-model ensemble (matches v7's partial-ensemble peak composition).
    models_spec = [
        ("lstm", 1337,  0.0),
        ("lstm", 2024,  0.0),
        ("lstm", 4242,  0.0),
        ("lstm",  777,  0.0),
        ("lstm", 31337, 0.0),
        ("gru",  2718,  0.0),
        ("lstm", 1337,  0.3),
    ]

    preds_test_all = []
    preds_val_all = []
    val_best_list = []
    Yv = tgt_anom_n[val_mask]   # note: compared in anomaly-norm space
    t_ids_val = t_ids[val_mask]

    for rnn_type, seed, pd_ in models_spec:
        pt, pv, bv = train_one(
            seed, train_ds, val_ds, test_ds, n_stations,
            epochs=150, pair_dropout=pd_, rnn_type=rnn_type,
            tta_passes=3, tta_sigma=0.02,
        )
        preds_test_all.append(pt)
        preds_val_all.append(pv)
        val_best_list.append((rnn_type, seed, pd_, bv))
        print(f"[{rnn_type} seed={seed} pd={pd_}] best val MSE (anomaly-norm): {bv:.5f}",
              flush=True)

        # Partial-ensemble save after each model (crash-safe).
        _pt_n = np.mean(preds_test_all, axis=0)
        _pv_n = np.mean(preds_val_all, axis=0)
        _pt = _pt_n * anom_std + climate[t_ids_te][:, None, :]
        _pv_raw_tgt = _pv_n * anom_std + climate[t_ids_val][:, None, :]
        _pv_raw_true = Ytgt[val_mask]
        _limits = {"temp": (-45.0, 55.0), "dewpoint": (-50.0, 35.0), "wind_speed": (0.0, 35.0)}
        for vi, v in enumerate(VARS):
            lo, hi = _limits[v]
            _pt[..., vi] = np.clip(_pt[..., vi], lo, hi)
            _pv_raw_tgt[..., vi] = np.clip(_pv_raw_tgt[..., vi], lo, hi)
        _mse = float(((_pv_raw_tgt - _pv_raw_true) ** 2).mean())
        _nrmse = math.sqrt(_mse) / math.sqrt(ref_var)
        _score = max(0.0, 1.0 - _nrmse)
        print(f"  >> partial ensemble ({len(preds_test_all)} models) val score: {_score:.4f}",
              flush=True)
        write_submission(test_df, _pt, OUT_PATH)

    # Final ensemble
    preds_test_n = np.mean(preds_test_all, axis=0)
    preds_val_n = np.mean(preds_val_all, axis=0)
    preds_test = preds_test_n * anom_std + climate[t_ids_te][:, None, :]
    preds_val = preds_val_n * anom_std + climate[t_ids_val][:, None, :]

    limits = {"temp": (-45.0, 55.0), "dewpoint": (-50.0, 35.0), "wind_speed": (0.0, 35.0)}
    for vi, v in enumerate(VARS):
        lo, hi = limits[v]
        preds_test[..., vi] = np.clip(preds_test[..., vi], lo, hi)
        preds_val[..., vi] = np.clip(preds_val[..., vi], lo, hi)

    Yv_raw = Ytgt[val_mask]
    mse = float(((preds_val - Yv_raw) ** 2).mean())
    nrmse = math.sqrt(mse) / math.sqrt(ref_var)
    score_est = max(0.0, 1.0 - nrmse)
    print("\n===== Ensemble validation estimate =====")
    for rnn_type, seed, pd_, bv in val_best_list:
        print(f"  [{rnn_type} seed={seed} pd={pd_}] best val (anomaly-norm MSE): {bv:.4f}")
    print(f"Val MSE raw: {mse:.4f}")
    print(f"Val nRMSE:   {nrmse:.4f}")
    print(f"Val score:   {score_est:.4f}")
    for vi, v in enumerate(VARS):
        mse_v = float(((preds_val[..., vi] - Yv_raw[..., vi]) ** 2).mean())
        print(f"  {v}: MSE = {mse_v:.4f}")

    write_submission(test_df, preds_test, OUT_PATH)


if __name__ == "__main__":
    main()
