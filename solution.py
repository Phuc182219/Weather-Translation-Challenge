"""
Weather Translation Challenge - Solution (v6)

Sequence-to-sequence translation of 72-hour weather observations between
weather stations, trained from scratch.

v6 builds on v5's climate-anomaly BiLSTM and adds techniques aimed at the
two unseen test station pairs (H->F, I->A) that cost us ~0.11 val->test
score on previous versions:

1. *Station-embedding dropout.* During training, with probability p the
   source or target station embedding is replaced by a learned "null"
   embedding. This forces the network to rely on the climate-anomaly
   features (which every station has in its own history) rather than
   memorising pair-specific transforms, so the model behaves more
   similarly on unseen pairs at test time.

2. *Within-pair mixup.* Two training samples from the same (source,
   target) pair are linearly interpolated with lambda ~ Beta(0.4, 0.4).
   Free regularisation that broadens the learned anomaly transformation.

3. *Heterogeneous ensemble.* Six BiLSTM seeds with different embedding
   dropouts and mixup intensities + two dilated-TCN seeds for
   architectural diversity.

4. *Test-time augmentation.* Each model is evaluated multiple times at
   test time with emb-dropout enabled, averaged.

Everything else is inherited from v5: climate-mean centering, global
anomaly-std normalisation, residual prediction, per-pair stratified val.
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
# Dataset (supports within-pair mixup)
# ---------------------------------------------------------------------------
class AnomalyDataset(Dataset):
    def __init__(self, src_anom_n, tgt_anom_n, s_ids, t_ids):
        self.src = src_anom_n
        self.tgt = tgt_anom_n
        self.s_ids = s_ids
        self.t_ids = t_ids
        # Build per-pair index lookup for mixup partners
        self.pair_idx = {}
        if src_anom_n is not None:
            keys = list(zip(s_ids.tolist(), t_ids.tolist()))
            for i, k in enumerate(keys):
                self.pair_idx.setdefault(k, []).append(i)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        sx = self.src[idx]
        ty = self.tgt[idx] if self.tgt is not None else np.zeros_like(sx)
        return (
            torch.from_numpy(sx),
            torch.from_numpy(ty),
            int(self.s_ids[idx]),
            int(self.t_ids[idx]),
            idx,
        )

    def mixup_partner(self, idx):
        """Return an index of another sample with the same (source, target) pair."""
        key = (int(self.s_ids[idx]), int(self.t_ids[idx]))
        pool = self.pair_idx.get(key, [idx])
        if len(pool) <= 1:
            return idx
        j = pool[np.random.randint(len(pool))]
        return j


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
# BiLSTM model with station-embedding dropout
# ---------------------------------------------------------------------------
class BiLSTMTranslator(nn.Module):
    def __init__(
        self,
        n_stations,
        d_model=192,
        n_layers=3,
        dropout=0.2,
        station_dim=48,
        emb_dropout=0.1,
    ):
        super().__init__()
        self.src_station_emb = nn.Embedding(n_stations, station_dim)
        self.tgt_station_emb = nn.Embedding(n_stations, station_dim)
        # Learned "null" embedding used during station-emb dropout + TTA.
        self.null_src_emb = nn.Parameter(torch.zeros(station_dim))
        self.null_tgt_emb = nn.Parameter(torch.zeros(station_dim))
        nn.init.trunc_normal_(self.null_src_emb, std=0.02)
        nn.init.trunc_normal_(self.null_tgt_emb, std=0.02)
        self.emb_dropout = emb_dropout

        in_feats = N_VARS + 4 + 2 * station_dim
        self.input_proj = nn.Linear(in_feats, d_model)

        self.rnn = nn.LSTM(
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

    def forward(self, x, s_ids, t_ids, force_emb_drop=None):
        """force_emb_drop: if set, overrides training-time dropout probability
        (used for TTA at inference to apply the null embedding a controlled
        fraction of the time)."""
        B = x.size(0)
        s_e = self.src_station_emb(s_ids)
        t_e = self.tgt_station_emb(t_ids)

        drop_p = force_emb_drop if force_emb_drop is not None else (
            self.emb_dropout if self.training else 0.0
        )
        if drop_p > 0:
            # Sample independent dropout mask per sample per role.
            m_s = (torch.rand(B, device=x.device) < drop_p).float().unsqueeze(-1)
            m_t = (torch.rand(B, device=x.device) < drop_p).float().unsqueeze(-1)
            s_e = s_e * (1 - m_s) + self.null_src_emb.unsqueeze(0) * m_s
            t_e = t_e * (1 - m_t) + self.null_tgt_emb.unsqueeze(0) * m_t

        station_feat = torch.cat([s_e, t_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        tf = self.time_feats.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, tf, station_feat], dim=-1)
        h = self.input_proj(tokens)
        h, _ = self.rnn(h)
        h = self.norm(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# Dilated TCN model (for ensemble diversity)
# ---------------------------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, k, dilation, dropout):
        super().__init__()
        pad = (k - 1) * dilation // 2  # centred padding for a non-causal 1D conv
        self.conv1 = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(c_out, c_out, k, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.do = nn.Dropout(dropout)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.act(y)
        y = self.do(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.act(y)
        y = self.do(y)
        return y + self.skip(x)


class TCNTranslator(nn.Module):
    def __init__(self, n_stations, d_model=160, n_blocks=4, k=5, dropout=0.15,
                 station_dim=48, emb_dropout=0.1):
        super().__init__()
        self.src_station_emb = nn.Embedding(n_stations, station_dim)
        self.tgt_station_emb = nn.Embedding(n_stations, station_dim)
        self.null_src_emb = nn.Parameter(torch.zeros(station_dim))
        self.null_tgt_emb = nn.Parameter(torch.zeros(station_dim))
        nn.init.trunc_normal_(self.null_src_emb, std=0.02)
        nn.init.trunc_normal_(self.null_tgt_emb, std=0.02)
        self.emb_dropout = emb_dropout

        in_feats = N_VARS + 4 + 2 * station_dim
        self.input_proj = nn.Conv1d(in_feats, d_model, 1)

        blocks = []
        for i in range(n_blocks):
            blocks.append(TemporalBlock(d_model, d_model, k, 2 ** i, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv1d(d_model, N_VARS, 1)
        self.register_buffer("time_feats", sinusoidal_time_feats(T))

    def forward(self, x, s_ids, t_ids, force_emb_drop=None):
        B = x.size(0)
        s_e = self.src_station_emb(s_ids)
        t_e = self.tgt_station_emb(t_ids)
        drop_p = force_emb_drop if force_emb_drop is not None else (
            self.emb_dropout if self.training else 0.0
        )
        if drop_p > 0:
            m_s = (torch.rand(B, device=x.device) < drop_p).float().unsqueeze(-1)
            m_t = (torch.rand(B, device=x.device) < drop_p).float().unsqueeze(-1)
            s_e = s_e * (1 - m_s) + self.null_src_emb.unsqueeze(0) * m_s
            t_e = t_e * (1 - m_t) + self.null_tgt_emb.unsqueeze(0) * m_t

        station_feat = torch.cat([s_e, t_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        tf = self.time_feats.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, tf, station_feat], dim=-1)  # (B, T, F)
        tokens = tokens.transpose(1, 2)  # (B, F, T) for Conv1d
        h = self.input_proj(tokens)
        h = self.blocks(h)
        out = self.head(h)  # (B, V, T)
        return out.transpose(1, 2)  # (B, T, V)


# ---------------------------------------------------------------------------
# Training one model
# ---------------------------------------------------------------------------
def train_one(
    seed,
    arch,
    train_ds,
    val_ds,
    test_ds,
    n_stations,
    epochs=120,
    mixup_alpha=0.4,
    emb_dropout=0.1,
):
    set_seed(seed)
    BATCH = 128
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    if arch == "lstm":
        model = BiLSTMTranslator(n_stations=n_stations, emb_dropout=emb_dropout).to(DEVICE)
        LR = 2e-3
    elif arch == "tcn":
        model = TCNTranslator(n_stations=n_stations, emb_dropout=emb_dropout).to(DEVICE)
        LR = 2.5e-3
    else:
        raise ValueError(arch)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{arch} seed={seed} drop={emb_dropout} mix={mixup_alpha}] "
          f"Model params: {n_params:,}", flush=True)

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
        for sx, ty, s_id, t_id, idx in train_loader:
            sx, ty = sx.to(DEVICE), ty.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)

            # Within-pair mixup: pair each sample with another sample from
            # the same (source, target) pair and blend.
            if mixup_alpha > 0 and random.random() < 0.5:
                # Sample lambda per-sample from Beta(alpha, alpha)
                lam = np.random.beta(mixup_alpha, mixup_alpha, size=sx.size(0)).astype(np.float32)
                lam = np.maximum(lam, 1 - lam)  # keep lam in [0.5, 1] (standard mixup trick)
                partner_idx = np.array(
                    [train_ds.mixup_partner(int(i)) for i in idx.numpy()], dtype=np.int64
                )
                sx2 = torch.from_numpy(train_ds.src[partner_idx]).to(DEVICE)
                ty2 = torch.from_numpy(train_ds.tgt[partner_idx]).to(DEVICE)
                lam_t = torch.from_numpy(lam).to(DEVICE).view(-1, 1, 1)
                sx = sx * lam_t + sx2 * (1 - lam_t)
                ty = ty * lam_t + ty2 * (1 - lam_t)

            for g in opt.param_groups:
                g["lr"] = LR * lr_at(step)

            delta = model(sx, s_id, t_id)
            pred = sx + delta
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
                for sx, ty, s_id, t_id, _idx in val_loader:
                    sx, ty = sx.to(DEVICE), ty.to(DEVICE)
                    s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
                    pred = sx + model(sx, s_id, t_id)
                    vl += F.mse_loss(pred, ty, reduction="mean").item() * sx.size(0)
                    vn += sx.size(0)
            vl /= max(1, vn)
            improved = vl < best_val - 1e-6
            flag = "*" if improved else " "
            if ep == 1 or ep % 10 == 0 or improved:
                print(
                    f"[{arch} seed={seed}] Ep {ep:3d}  lr={opt.param_groups[0]['lr']:.2e}  "
                    f"train={tr_loss:.4f}  val={vl:.4f} {flag}  elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
            if improved:
                best_val = vl
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            if ep == 1 or ep % 10 == 0 or ep == epochs:
                print(
                    f"[{arch} seed={seed}] Ep {ep:3d}  lr={opt.param_groups[0]['lr']:.2e}  "
                    f"train={tr_loss:.4f}  elapsed={time.time()-t0:.1f}s",
                    flush=True,
                )
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    # ------- Deterministic inference (no emb dropout) + TTA mean --------
    def predict(loader, tta_passes=1, tta_drop=0.0):
        preds = []
        with torch.no_grad():
            for sx, _ty, s_id, t_id, _idx in loader:
                sx = sx.to(DEVICE)
                s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
                accum = sx + model(sx, s_id, t_id)
                for _ in range(tta_passes - 1):
                    accum = accum + (sx + model(sx, s_id, t_id, force_emb_drop=tta_drop))
                accum = accum / tta_passes
                preds.append(accum.cpu().numpy())
        return np.concatenate(preds, axis=0)

    preds_te = predict(test_loader, tta_passes=4, tta_drop=0.05)
    preds_val = predict(val_loader, tta_passes=4, tta_drop=0.05) if val_loader is not None else None
    return preds_te, preds_val, best_val


# ---------------------------------------------------------------------------
# Submission
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

    src_anom = Xsrc - climate[s_ids][:, None, :]
    tgt_anom = Ytgt - climate[t_ids][:, None, :]
    src_anom_n = (src_anom / (anom_std + 1e-6)).astype(np.float32)
    tgt_anom_n = (tgt_anom / (anom_std + 1e-6)).astype(np.float32)

    Xte = reshape_wide_to_series(test_df, "source")
    s_ids_te = np.array([stn_to_id[s] for s in test_df["source_city"]], dtype=np.int64)
    t_ids_te = np.array([stn_to_id[s] for s in test_df["target_city"]], dtype=np.int64)
    src_anom_te = ((Xte - climate[s_ids_te][:, None, :]) / (anom_std + 1e-6)).astype(np.float32)

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

    train_ds = AnomalyDataset(src_anom_n[~val_mask], tgt_anom_n[~val_mask],
                               s_ids[~val_mask], t_ids[~val_mask])
    val_ds = AnomalyDataset(src_anom_n[val_mask], tgt_anom_n[val_mask],
                             s_ids[val_mask], t_ids[val_mask])
    test_ds = AnomalyDataset(src_anom_te, None, s_ids_te, t_ids_te)
    print(f"Train/Val/Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    ref_var = float(Ytgt.var())
    print(f"Ref var (proxy): {ref_var:.3f}")

    # Ensemble: 6 LSTM seeds with different emb_dropout/mixup settings for
    # diversity, plus 2 TCN seeds.
    models_spec = [
        ("lstm", 1337,  0.10, 0.4),
        ("lstm", 2024,  0.10, 0.4),
        ("lstm", 4242,  0.15, 0.3),
        ("lstm",  777,  0.05, 0.4),
        ("lstm", 31337, 0.10, 0.5),
        ("lstm", 2718,  0.15, 0.2),
        ("tcn",   1337, 0.10, 0.3),
        ("tcn",   4242, 0.10, 0.4),
    ]

    preds_test_all = []
    preds_val_all = []
    val_best_list = []
    Yv = Ytgt[val_mask]
    t_ids_val = t_ids[val_mask]

    for arch, seed, emb_do, mix_a in models_spec:
        pt, pv, bv = train_one(
            seed, arch, train_ds, val_ds, test_ds, n_stations,
            epochs=120, emb_dropout=emb_do, mixup_alpha=mix_a,
        )
        preds_test_all.append(pt)
        preds_val_all.append(pv)
        val_best_list.append((arch, seed, bv))
        print(f"[{arch} seed={seed}] best val MSE (anomaly-norm): {bv:.5f}", flush=True)

        # Save partial-ensemble submission after each model (crash-safe).
        _pt_n = np.mean(preds_test_all, axis=0)
        _pv_n = np.mean(preds_val_all, axis=0)
        _pt = _pt_n * anom_std + climate[t_ids_te][:, None, :]
        _pv = _pv_n * anom_std + climate[t_ids_val][:, None, :]
        _limits = {"temp": (-45.0, 55.0), "dewpoint": (-50.0, 35.0), "wind_speed": (0.0, 35.0)}
        for vi, v in enumerate(VARS):
            lo, hi = _limits[v]
            _pt[..., vi] = np.clip(_pt[..., vi], lo, hi)
            _pv[..., vi] = np.clip(_pv[..., vi], lo, hi)
        _mse = float(((_pv - Yv) ** 2).mean())
        _nrmse = math.sqrt(_mse) / math.sqrt(ref_var)
        _score = max(0.0, 1.0 - _nrmse)
        print(f"  >> partial ensemble ({len(preds_test_all)} models) val score: {_score:.4f}", flush=True)
        write_submission(test_df, _pt, OUT_PATH)

    # Final ensemble
    preds_test_n = np.mean(preds_test_all, axis=0)
    preds_val_n = np.mean(preds_val_all, axis=0)
    preds_test = preds_test_n * anom_std + climate[t_ids_te][:, None, :]
    preds_val  = preds_val_n  * anom_std + climate[t_ids_val][:, None, :]

    limits = {"temp": (-45.0, 55.0), "dewpoint": (-50.0, 35.0), "wind_speed": (0.0, 35.0)}
    for vi, v in enumerate(VARS):
        lo, hi = limits[v]
        preds_test[..., vi] = np.clip(preds_test[..., vi], lo, hi)
        preds_val[..., vi] = np.clip(preds_val[..., vi], lo, hi)

    mse = float(((preds_val - Yv) ** 2).mean())
    nrmse = math.sqrt(mse) / math.sqrt(ref_var)
    score_est = max(0.0, 1.0 - nrmse)
    print("\n===== Ensemble validation estimate =====")
    for arch, seed, bv in val_best_list:
        print(f"  [{arch} seed={seed}] best val (anomaly-norm MSE): {bv:.4f}")
    print(f"Val MSE raw: {mse:.4f}")
    print(f"Val nRMSE:   {nrmse:.4f}")
    print(f"Val score:   {score_est:.4f}")
    for vi, v in enumerate(VARS):
        mse_v = float(((preds_val[..., vi] - Yv[..., vi]) ** 2).mean())
        print(f"  {v}: MSE = {mse_v:.4f}")

    write_submission(test_df, preds_test, OUT_PATH)


if __name__ == "__main__":
    main()
