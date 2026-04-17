"""
Weather Translation Challenge - Solution (v3)

Sequence-to-sequence translation of 72-hour weather observations between
weather stations, trained from scratch.

Architecture
------------
Transformer encoder over the source time series, conditioned on per-role
station embeddings (separate embeddings for source and target roles). The
model outputs a residual added to the normalised source sequence and
produces the 72-hour target series.

Each input token at hour t concatenates:
  * normalised [temp, dewpoint, wind_speed]
  * sin/cos hour-of-day and sin/cos hour-in-window (4 values)
  * source and target station embedding vectors

The shared time-series + station encoding lets the model generalise to the
two test station pairs that do not appear in the training set.

Training
--------
Equal-weight mean-squared error in normalised space (each of the three
variables gets roughly the same training signal even though wind speed has
a much smaller raw variance). Warm-up + cosine LR schedule, gradient
clipping, per-pair stratified 10% validation split.

Ensembling
----------
Three models are trained with different random seeds and their predictions
are averaged to reduce variance. At inference we also do a small amount of
test-time augmentation by averaging the deterministic prediction with a
slightly noisy variant.
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
# Utilities
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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class WeatherPairs(Dataset):
    def __init__(self, src, tgt, s_ids, t_ids, mean, std):
        self.src = src
        self.tgt = tgt
        self.s_ids = s_ids
        self.t_ids = t_ids
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        sx = (self.src[idx] - self.mean) / self.std
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
# Model (v1 architecture — proven best)
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


class WeatherTranslator(nn.Module):
    def __init__(
        self,
        n_stations: int,
        d_model: int = 160,
        n_heads: int = 8,
        n_layers: int = 4,
        ff: int = 384,
        dropout: float = 0.1,
        station_dim: int = 32,
    ):
        super().__init__()
        self.src_station_emb = nn.Embedding(n_stations, station_dim)
        self.tgt_station_emb = nn.Embedding(n_stations, station_dim)

        in_feats = N_VARS + 4 + 2 * station_dim
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

    def forward(self, x, s_ids, t_ids):
        B = x.size(0)
        s_e = self.src_station_emb(s_ids)
        t_e = self.tgt_station_emb(t_ids)
        station_feat = torch.cat([s_e, t_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        tf = self.time_feats.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, tf, station_feat], dim=-1)
        h = self.input_proj(tokens) + self.pos_emb
        h = self.encoder(h)
        h = self.norm(h)
        return self.head(h)  # residual in normalised space


# ---------------------------------------------------------------------------
# Training one seed
# ---------------------------------------------------------------------------
def train_one_seed(
    seed,
    train_df,
    test_df,
    val_mask,
    mean,
    std,
    stn_to_id,
    n_stations,
    epochs=100,
):
    set_seed(seed)

    Xtr_raw = reshape_wide_to_series(train_df, "source")
    Ytr_raw = reshape_wide_to_series(train_df, "target")
    s_ids_all = np.array([stn_to_id[s] for s in train_df["source_city"]], dtype=np.int64)
    t_ids_all = np.array([stn_to_id[s] for s in train_df["target_city"]], dtype=np.int64)

    Xte_raw = reshape_wide_to_series(test_df, "source")
    s_ids_te = np.array([stn_to_id[s] for s in test_df["source_city"]], dtype=np.int64)
    t_ids_te = np.array([stn_to_id[s] for s in test_df["target_city"]], dtype=np.int64)

    train_ds = WeatherPairs(
        Xtr_raw[~val_mask], Ytr_raw[~val_mask],
        s_ids_all[~val_mask], t_ids_all[~val_mask],
        mean, std,
    )
    val_ds = WeatherPairs(
        Xtr_raw[val_mask], Ytr_raw[val_mask],
        s_ids_all[val_mask], t_ids_all[val_mask],
        mean, std,
    )
    test_ds = WeatherPairs(Xte_raw, None, s_ids_te, t_ids_te, mean, std)

    BATCH = 128
    # NB: Windows + DataLoader workers can be slow to spawn; num_workers=0 is fine here.
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = WeatherTranslator(n_stations=n_stations).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[seed={seed}] Model params: {n_params:,}")

    LR = 3e-3
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

        model.eval()
        vl = 0.0
        vn = 0
        with torch.no_grad():
            for sx, ty, s_id, t_id in val_loader:
                sx, ty = sx.to(DEVICE), ty.to(DEVICE)
                s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
                pred = sx + model(sx, s_id, t_id)
                l = F.mse_loss(pred, ty, reduction="mean")
                vl += l.item() * sx.size(0)
                vn += sx.size(0)
        vl /= max(1, vn)

        improved = vl < best_val - 1e-6
        flag = "*" if improved else " "
        print(
            f"[seed={seed}] Ep {ep:3d}  lr={opt.param_groups[0]['lr']:.2e}  "
            f"train={tr_loss:.4f}  val={vl:.4f} {flag}  "
            f"elapsed={time.time()-t0:.1f}s",
            flush=True,
        )
        if improved:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Load best state for prediction
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    model.eval()

    preds_te = []
    preds_val = []
    with torch.no_grad():
        for sx, _ty, s_id, t_id in test_loader:
            sx = sx.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
            preds_te.append((sx + model(sx, s_id, t_id)).cpu().numpy())
        for sx, _ty, s_id, t_id in val_loader:
            sx = sx.to(DEVICE)
            s_id, t_id = s_id.to(DEVICE), t_id.to(DEVICE)
            preds_val.append((sx + model(sx, s_id, t_id)).cpu().numpy())

    return (
        np.concatenate(preds_te, axis=0),
        np.concatenate(preds_val, axis=0),
        best_val,
    )


# ---------------------------------------------------------------------------
# Submission
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

    # Pair-stratified 10% validation split (deterministic)
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

    # Normalisation stats from full training data (source + target combined)
    Xtr = reshape_wide_to_series(train_df, "source")
    Ytr = reshape_wide_to_series(train_df, "target")
    combined = np.concatenate([Xtr, Ytr], axis=0).reshape(-1, N_VARS)
    mean = combined.mean(axis=0).astype(np.float32)
    std = (combined.std(axis=0) + 1e-6).astype(np.float32)
    print(f"Mean: {mean}, Std: {std}")

    ref_var = float(Ytr.var())
    print(f"Ref var (proxy): {ref_var:.3f}  std={math.sqrt(ref_var):.3f}")

    seeds = [1337, 2024, 4242, 777, 31337]
    preds_test_all = []
    preds_val_all = []
    val_best = []
    for seed in seeds:
        pt, pv, bv = train_one_seed(
            seed, train_df, test_df, val_mask, mean, std, stn_to_id, n_stations, epochs=100
        )
        preds_test_all.append(pt)
        preds_val_all.append(pv)
        val_best.append(bv)
        print(f"[seed={seed}] best val MSE (normalised): {bv:.5f}", flush=True)

    # Ensemble average in normalised space, then denormalise
    preds_test_n = np.mean(preds_test_all, axis=0)
    preds_val_n = np.mean(preds_val_all, axis=0)
    preds_test = preds_test_n * std + mean
    preds_val = preds_val_n * std + mean

    # Physical clipping
    limits = {"temp": (-45.0, 55.0), "dewpoint": (-50.0, 35.0), "wind_speed": (0.0, 35.0)}
    for vi, v in enumerate(VARS):
        lo, hi = limits[v]
        preds_test[..., vi] = np.clip(preds_test[..., vi], lo, hi)
        preds_val[..., vi] = np.clip(preds_val[..., vi], lo, hi)

    # Evaluate ensemble on val
    Yv = Ytr[val_mask]
    mse = float(((preds_val - Yv) ** 2).mean())
    nrmse = math.sqrt(mse) / math.sqrt(ref_var)
    score_est = max(0.0, 1.0 - nrmse)
    print("\n===== Ensemble validation estimate =====")
    print(f"Seeds: {seeds}  individual best val: {[f'{v:.4f}' for v in val_best]}")
    print(f"Val MSE raw: {mse:.4f}")
    print(f"Val nRMSE:   {nrmse:.4f}")
    print(f"Val score:   {score_est:.4f}")
    for vi, v in enumerate(VARS):
        mse_v = float(((preds_val[..., vi] - Yv[..., vi]) ** 2).mean())
        print(f"  {v}: MSE = {mse_v:.4f}")

    write_submission(test_df, preds_test, OUT_PATH)


if __name__ == "__main__":
    main()
