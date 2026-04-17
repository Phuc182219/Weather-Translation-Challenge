"""
Weather Translation Challenge - Solution (v2)

Sequence-to-sequence translation of 72-hour weather observations between
weather stations, trained from scratch.

Key ideas
---------
* Transformer encoder over the source time series, conditioned on shared
  station embeddings used for both source and target. This lets the model
  generalise to station pairs unseen in training (2 of the 14 test pairs).
* Residual prediction: the model outputs delta added to the normalised source.
* Input features at each hour: [temp, dewpoint, wind_speed, derived relative
  humidity] + sin/cos hour-of-day + sin/cos hour-of-window, concatenated with
  source and target station embeddings.
* Variance-weighted loss in normalised space - effectively minimises raw-unit
  MSE which matches the challenge scoring (pooled nRMSE across 216 values).
* EMA weights and a small multi-seed ensemble for a robust final submission.
"""

import math
import os
import random
import time
from copy import deepcopy

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
    return np.stack(arrs, axis=-1)  # (N, T, V)


def stations_list(train_df: pd.DataFrame, test_df: pd.DataFrame):
    s = set(train_df["source_city"]).union(train_df["target_city"])
    s = s.union(test_df["source_city"]).union(test_df["target_city"])
    return sorted(s)


def rel_humidity(temp_c: np.ndarray, dew_c: np.ndarray) -> np.ndarray:
    """August-Roche-Magnus approx. Returns RH in [0, 120]ish."""
    a, b = 17.625, 243.04
    num = np.exp((a * dew_c) / (b + dew_c))
    den = np.exp((a * temp_c) / (b + temp_c))
    rh = 100.0 * num / np.clip(den, 1e-6, None)
    return np.clip(rh, 0.0, 150.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WeatherPairs(Dataset):
    def __init__(self, src, tgt, s_ids, t_ids, mean, std, noise_std=0.0):
        self.src = src
        self.tgt = tgt
        self.s_ids = s_ids
        self.t_ids = t_ids
        self.mean = mean
        self.std = std
        self.noise_std = noise_std

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        sx = (self.src[idx] - self.mean) / self.std
        if self.noise_std > 0:
            sx = sx + np.random.normal(0.0, self.noise_std, size=sx.shape).astype(np.float32)
        if self.tgt is not None:
            ty = (self.tgt[idx] - self.mean) / self.std
        else:
            ty = np.zeros_like(sx)
        return (
            torch.from_numpy(sx),
            torch.from_numpy(ty),
            int(self.s_ids[idx]),
            int(self.t_ids[idx]),
        )


# ---------------------------------------------------------------------------
# Model
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
    )  # (T, 4)


class WeatherTranslator(nn.Module):
    def __init__(
        self,
        n_stations: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 5,
        ff: int = 512,
        dropout: float = 0.1,
        station_dim: int = 48,
    ):
        super().__init__()
        # Shared station embedding (used for both source and target roles).
        self.station_emb = nn.Embedding(n_stations, station_dim)
        # Small role-specific bias vectors added to the shared embedding.
        self.role_src = nn.Parameter(torch.zeros(station_dim))
        self.role_tgt = nn.Parameter(torch.zeros(station_dim))

        # Input = [3 vars normalised, rel humidity normalised, 4 time feats, src_emb, tgt_emb]
        in_feats = N_VARS + 1 + 4 + 2 * station_dim
        self.input_proj = nn.Linear(in_feats, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_VARS)

        self.register_buffer("time_feats", sinusoidal_time_feats(T))

    def forward(self, x, rh, s_ids, t_ids):
        """
        x:  (B, T, N_VARS) normalised source
        rh: (B, T, 1)      normalised humidity
        s_ids, t_ids: (B,) long
        """
        B = x.size(0)
        s_e = self.station_emb(s_ids) + self.role_src
        t_e = self.station_emb(t_ids) + self.role_tgt
        station_feat = torch.cat([s_e, t_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        tf = self.time_feats.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, rh, tf, station_feat], dim=-1)
        h = self.input_proj(tokens) + self.pos_emb
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h)  # (B, T, N_VARS) residual in normalised space


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def load_into(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


# ---------------------------------------------------------------------------
# Train a single model (one seed)
# ---------------------------------------------------------------------------
def train_one(seed, train_df, test_df, val_mask, mean, std, rh_mean, rh_std, epochs=90):
    set_seed(seed)

    stations = stations_list(train_df, test_df)
    stn_to_id = {s: i for i, s in enumerate(stations)}
    n_stations = len(stations)

    # (N, T, V) arrays
    Xtr_raw = reshape_wide_to_series(train_df, "source")
    Ytr_raw = reshape_wide_to_series(train_df, "target")
    s_ids_all = np.array([stn_to_id[s] for s in train_df["source_city"]], dtype=np.int64)
    t_ids_all = np.array([stn_to_id[s] for s in train_df["target_city"]], dtype=np.int64)

    Xte_raw = reshape_wide_to_series(test_df, "source")
    s_ids_te = np.array([stn_to_id[s] for s in test_df["source_city"]], dtype=np.int64)
    t_ids_te = np.array([stn_to_id[s] for s in test_df["target_city"]], dtype=np.int64)

    # Precompute relative humidity
    rh_tr = rel_humidity(Xtr_raw[..., 0], Xtr_raw[..., 1])  # (N, T)
    rh_te = rel_humidity(Xte_raw[..., 0], Xte_raw[..., 1])

    def normalize_rh(rh):
        return (rh - rh_mean) / (rh_std + 1e-6)

    rh_tr_n = normalize_rh(rh_tr).astype(np.float32)
    rh_te_n = normalize_rh(rh_te).astype(np.float32)

    train_ds = WeatherPairs(
        Xtr_raw[~val_mask], Ytr_raw[~val_mask],
        s_ids_all[~val_mask], t_ids_all[~val_mask],
        mean, std, noise_std=0.02,
    )
    val_ds = WeatherPairs(
        Xtr_raw[val_mask], Ytr_raw[val_mask],
        s_ids_all[val_mask], t_ids_all[val_mask],
        mean, std, noise_std=0.0,
    )
    test_ds = WeatherPairs(Xte_raw, None, s_ids_te, t_ids_te, mean, std, noise_std=0.0)

    # We need to zip in the precomputed humidity. We'll use indices and custom collate.
    rh_train_arr = rh_tr_n[~val_mask]
    rh_val_arr = rh_tr_n[val_mask]

    def make_batches(ds, rh_arr, batch_size, shuffle):
        idx = np.arange(len(ds))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            b = idx[i : i + batch_size]
            # build batch
            sx = np.stack([(ds.src[j] - ds.mean) / ds.std for j in b])
            if ds.noise_std > 0:
                sx = sx + np.random.normal(0.0, ds.noise_std, size=sx.shape).astype(np.float32)
            if ds.tgt is not None:
                ty = np.stack([(ds.tgt[j] - ds.mean) / ds.std for j in b])
            else:
                ty = np.zeros_like(sx)
            rh_b = rh_arr[b][..., None]  # (B, T, 1)
            s_id = ds.s_ids[b]
            t_id = ds.t_ids[b]
            yield (
                torch.from_numpy(sx).float(),
                torch.from_numpy(ty).float(),
                torch.from_numpy(rh_b).float(),
                torch.from_numpy(s_id).long(),
                torch.from_numpy(t_id).long(),
            )

    BATCH = 128

    model = WeatherTranslator(n_stations=n_stations).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed={seed}] Model params: {n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    ema = EMA(model, decay=0.999)

    # Variance-weighted loss: weights proportional to per-var std^2 so the loss
    # matches raw-unit MSE (which matches the scoring metric which pools all 3
    # variables into one MSE).
    var_weights = torch.tensor((std / std.max()) ** 2, dtype=torch.float32, device=DEVICE)

    # Approximate steps for scheduler (recomputed each epoch with shuffle seeded)
    steps_per_epoch = max(1, math.ceil(len(train_ds) / BATCH))
    total_steps = epochs * steps_per_epoch
    warmup = steps_per_epoch * 3

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog))

    best_val = float("inf")
    step = 0
    t0 = time.time()
    best_state = None
    best_ema = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for sx, ty, rh_b, s_id, t_id in make_batches(train_ds, rh_train_arr, BATCH, shuffle=True):
            sx, ty, rh_b = sx.to(DEVICE), ty.to(DEVICE), rh_b.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)

            for g in opt.param_groups:
                g["lr"] = 3e-3 * lr_at(step)

            delta = model(sx, rh_b, s_id, t_id)
            pred = sx + delta
            err = (pred - ty) ** 2
            loss = (err * var_weights.view(1, 1, -1)).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update(model)

            tr_loss += loss.item() * sx.size(0)
            tr_n += sx.size(0)
            step += 1

        tr_loss /= max(1, tr_n)

        # Val using EMA weights (better estimate for deployment)
        eval_model = WeatherTranslator(n_stations=n_stations).to(DEVICE)
        ema.load_into(eval_model)
        eval_model.eval()
        vl = 0.0
        vn = 0
        with torch.no_grad():
            for sx, ty, rh_b, s_id, t_id in make_batches(val_ds, rh_val_arr, 256, shuffle=False):
                sx, ty, rh_b = sx.to(DEVICE), ty.to(DEVICE), rh_b.to(DEVICE)
                s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
                pred = sx + eval_model(sx, rh_b, s_id, t_id)
                err = (pred - ty) ** 2
                l = (err * var_weights.view(1, 1, -1)).mean()
                vl += l.item() * sx.size(0)
                vn += sx.size(0)
        vl /= max(1, vn)

        improved = vl < best_val - 1e-6
        flag = "*" if improved else " "
        print(
            f"[seed={seed}] Ep {ep:3d}  lr={opt.param_groups[0]['lr']:.2e}  "
            f"train={tr_loss:.4f}  val={vl:.4f} {flag}  "
            f"elapsed={time.time()-t0:.1f}s"
        )
        if improved:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ema = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}

    # Predict test with EMA weights at the end of training (no early stopping;
    # cosine decay handles regularisation)
    final_model = WeatherTranslator(n_stations=n_stations).to(DEVICE)
    ema.load_into(final_model)
    final_model.eval()

    preds_te = []
    preds_val = []
    with torch.no_grad():
        for sx, _ty, rh_b, s_id, t_id in make_batches(test_ds, rh_te_n, 256, shuffle=False):
            sx, rh_b = sx.to(DEVICE), rh_b.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
            pred = sx + final_model(sx, rh_b, s_id, t_id)
            preds_te.append(pred.cpu().numpy())
        for sx, ty, rh_b, s_id, t_id in make_batches(val_ds, rh_val_arr, 256, shuffle=False):
            sx, ty, rh_b = sx.to(DEVICE), ty.to(DEVICE), rh_b.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
            pred = sx + final_model(sx, rh_b, s_id, t_id)
            preds_val.append(pred.cpu().numpy())

    preds_te = np.concatenate(preds_te, axis=0)
    preds_val = np.concatenate(preds_val, axis=0)
    return preds_te, preds_val, best_val


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------
def write_submission(test_df: pd.DataFrame, preds: np.ndarray, out_path: str):
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
# Entry point
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")

    # Build a pair-stratified validation split
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

    # Compute normalisation from FULL training set (including val split - stats only,
    # no leakage since we never see the val labels during training).
    Xtr = reshape_wide_to_series(train_df, "source")
    Ytr = reshape_wide_to_series(train_df, "target")
    combined = np.concatenate([Xtr, Ytr], axis=0).reshape(-1, N_VARS)
    mean = combined.mean(axis=0).astype(np.float32)
    std = (combined.std(axis=0) + 1e-6).astype(np.float32)
    print(f"Mean: {mean}, Std: {std}")

    rh_all_src = rel_humidity(Xtr[..., 0], Xtr[..., 1])
    rh_all_tgt = rel_humidity(Ytr[..., 0], Ytr[..., 1])
    rh_mean = float(np.concatenate([rh_all_src, rh_all_tgt]).mean())
    rh_std = float(np.concatenate([rh_all_src, rh_all_tgt]).std())
    print(f"RH mean/std: {rh_mean:.2f} / {rh_std:.2f}")

    # Also compute the variance used by the metric over all training targets
    ref_var = float(Ytr.var())
    print(f"Ref var (proxy): {ref_var:.3f}  std={math.sqrt(ref_var):.3f}")

    # Train 3 seeds and average
    seeds = [1337, 2024, 4242]
    preds_test_list = []
    preds_val_list = []
    val_losses = []
    for seed in seeds:
        pt, pv, bv = train_one(
            seed, train_df, test_df, val_mask, mean, std, rh_mean, rh_std, epochs=90
        )
        preds_test_list.append(pt)
        preds_val_list.append(pv)
        val_losses.append(bv)
        print(f"[seed={seed}] best val variance-weighted MSE: {bv:.5f}")

    preds_test_n = np.mean(preds_test_list, axis=0)   # normalised
    preds_val_n = np.mean(preds_val_list, axis=0)     # normalised

    # Denormalise
    preds_test = preds_test_n * std + mean
    preds_val = preds_val_n * std + mean

    # Physical clipping
    limits = {"temp": (-45.0, 55.0), "dewpoint": (-50.0, 35.0), "wind_speed": (0.0, 35.0)}
    for vi, v in enumerate(VARS):
        lo, hi = limits[v]
        preds_test[..., vi] = np.clip(preds_test[..., vi], lo, hi)
        preds_val[..., vi] = np.clip(preds_val[..., vi], lo, hi)

    # Report validation score using ensemble
    Yv = Ytr[val_mask]
    mse = float(((preds_val - Yv) ** 2).mean())
    ref_var_val = float(Ytr.var())  # proxy for pool variance
    nrmse = math.sqrt(mse) / math.sqrt(ref_var_val)
    score_est = max(0.0, 1.0 - nrmse)
    print("\n===== Ensemble validation estimate =====")
    print(f"Val MSE raw: {mse:.4f}")
    print(f"Val nRMSE:   {nrmse:.4f}")
    print(f"Val score:   {score_est:.4f}")
    for vi, v in enumerate(VARS):
        mse_v = float(((preds_val[..., vi] - Yv[..., vi]) ** 2).mean())
        print(f"  {v}: MSE = {mse_v:.4f}")

    write_submission(test_df, preds_test, OUT_PATH)


if __name__ == "__main__":
    main()
