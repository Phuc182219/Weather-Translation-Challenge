"""
Weather Translation Challenge - v1
Seq2seq translation of 72-hour weather observations between station pairs.

Model: Transformer encoder conditioned on source/target station embeddings.
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
# Paths and constants
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "public")
if not os.path.exists(os.path.join(DATA_DIR, "train.csv")):
    # Submission container layout: data at /kaggle/input/dataset/public/
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
T = 72  # hours per sample
N_VARS = len(VARS)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def reshape_wide_to_series(df: pd.DataFrame, role: str) -> np.ndarray:
    """Return (N, T, 3) array for role in {'source', 'target'}."""
    arrs = []
    for v in VARS:
        cols = [f"{role}_{v}_{i}" for i in range(T)]
        arrs.append(df[cols].values.astype(np.float32))
    # stack into (N, T, V)
    return np.stack(arrs, axis=-1)


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
# Model
# ---------------------------------------------------------------------------
def sinusoidal_time_feats(T: int) -> torch.Tensor:
    """Return (T, 4) features: daily sin/cos + multi-day sin/cos."""
    hours = torch.arange(T, dtype=torch.float32)
    feats = torch.stack(
        [
            torch.sin(2 * math.pi * hours / 24.0),
            torch.cos(2 * math.pi * hours / 24.0),
            torch.sin(2 * math.pi * hours / 72.0),
            torch.cos(2 * math.pi * hours / 72.0),
        ],
        dim=-1,
    )
    return feats  # (T, 4)


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
        self.n_stations = n_stations
        self.d_model = d_model

        self.src_station_emb = nn.Embedding(n_stations, station_dim)
        self.tgt_station_emb = nn.Embedding(n_stations, station_dim)

        in_feats = N_VARS + 4 + 2 * station_dim  # vars + time feats + src/tgt embs
        self.input_proj = nn.Linear(in_feats, d_model)

        self.pos_emb = nn.Parameter(torch.zeros(1, T, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, N_VARS)

        # Register a constant time feature buffer
        self.register_buffer("time_feats", sinusoidal_time_feats(T))  # (T, 4)

    def forward(self, x, s_ids, t_ids):
        """
        x: (B, T, N_VARS) normalised source series
        s_ids, t_ids: (B,) long
        returns: (B, T, N_VARS) residual prediction (delta from source)
        """
        B = x.size(0)
        s_e = self.src_station_emb(s_ids)  # (B, D)
        t_e = self.tgt_station_emb(t_ids)  # (B, D)
        station_feat = torch.cat([s_e, t_e], dim=-1).unsqueeze(1).expand(-1, T, -1)
        tf = self.time_feats.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([x, tf, station_feat], dim=-1)  # (B, T, F)
        h = self.input_proj(tokens) + self.pos_emb
        h = self.encoder(h)
        h = self.norm(h)
        delta = self.head(h)
        return delta  # residual: final prediction = x + delta (in normalised space)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def variance_weighted_mse(pred, tgt, weights):
    """pred, tgt: (B, T, V); weights: (V,) - loss weighted so each var contributes equally."""
    err = (pred - tgt) ** 2  # (B, T, V)
    w = weights.view(1, 1, -1)
    return (err * w).mean()


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame, model_path: str):
    # Map stations to integer ids
    stations = stations_list(train_df, test_df)
    stn_to_id = {s: i for i, s in enumerate(stations)}
    n_stations = len(stations)
    print(f"Stations: {stations}")

    # Wide -> (N, T, V)
    Xtr = reshape_wide_to_series(train_df, "source")
    Ytr = reshape_wide_to_series(train_df, "target")
    s_ids_tr = np.array([stn_to_id[s] for s in train_df["source_city"]], dtype=np.int64)
    t_ids_tr = np.array([stn_to_id[s] for s in train_df["target_city"]], dtype=np.int64)

    Xte = reshape_wide_to_series(test_df, "source")
    s_ids_te = np.array([stn_to_id[s] for s in test_df["source_city"]], dtype=np.int64)
    t_ids_te = np.array([stn_to_id[s] for s in test_df["target_city"]], dtype=np.int64)

    # Per-variable normalisation (from train: both source and target values combined)
    combined = np.concatenate([Xtr, Ytr], axis=0).reshape(-1, N_VARS)
    mean = combined.mean(axis=0)
    std = combined.std(axis=0) + 1e-6
    print(f"Mean: {mean}, Std: {std}")

    # Train/val split stratified by pair so every pair is represented in val
    rng = np.random.default_rng(SEED)
    pair_keys = train_df["source_city"] + "->" + train_df["target_city"]
    val_mask = np.zeros(len(train_df), dtype=bool)
    for _, grp in train_df.groupby(pair_keys):
        idx = grp.index.to_numpy()
        take = max(1, int(0.1 * len(idx)))
        chosen = rng.choice(idx, size=take, replace=False)
        val_mask[chosen] = True

    train_ds = WeatherPairs(
        Xtr[~val_mask], Ytr[~val_mask], s_ids_tr[~val_mask], t_ids_tr[~val_mask], mean, std
    )
    val_ds = WeatherPairs(
        Xtr[val_mask], Ytr[val_mask], s_ids_tr[val_mask], t_ids_tr[val_mask], mean, std
    )
    test_ds = WeatherPairs(Xte, None, s_ids_te, t_ids_te, mean, std)

    print(f"Train/Val/Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")

    BATCH = 128
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = WeatherTranslator(n_stations=n_stations).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")

    EPOCHS = 80
    LR = 3e-3
    WD = 1e-4

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = EPOCHS * steps_per_epoch
    warmup = steps_per_epoch * 3

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    # Weights to make each variable contribute equally to the training loss
    # (since their normalised scales are similar, use equal weights = 1)
    var_weights = torch.ones(N_VARS, device=DEVICE)

    best_val = float("inf")
    best_epoch = -1
    patience = 0
    PATIENCE_MAX = 15
    step = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for sx, ty, s_id, t_id in train_loader:
            sx = sx.to(DEVICE, non_blocking=True)
            ty = ty.to(DEVICE, non_blocking=True)
            s_id = s_id.to(DEVICE)
            t_id = t_id.to(DEVICE)

            for g in opt.param_groups:
                g["lr"] = LR * lr_at(step)

            delta = model(sx, s_id, t_id)
            pred = sx + delta
            loss = variance_weighted_mse(pred, ty, var_weights)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item() * sx.size(0)
            tr_n += sx.size(0)
            step += 1

        tr_loss /= max(1, tr_n)

        # Validation
        model.eval()
        vl = 0.0
        vn = 0
        with torch.no_grad():
            for sx, ty, s_id, t_id in val_loader:
                sx = sx.to(DEVICE)
                ty = ty.to(DEVICE)
                s_id = s_id.to(DEVICE)
                t_id = t_id.to(DEVICE)
                pred = sx + model(sx, s_id, t_id)
                l = F.mse_loss(pred, ty, reduction="mean")
                vl += l.item() * sx.size(0)
                vn += sx.size(0)
        vl /= max(1, vn)

        improved = vl < best_val - 1e-6
        flag = "*" if improved else " "
        print(
            f"Ep {ep:3d}  lr={opt.param_groups[0]['lr']:.2e}  "
            f"train={tr_loss:.4f}  val={vl:.4f} {flag}  "
            f"elapsed={time.time()-t0:.1f}s"
        )
        if improved:
            best_val = vl
            best_epoch = ep
            patience = 0
            torch.save(
                {
                    "state": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "stations": stations,
                },
                model_path,
            )
        else:
            patience += 1
            if patience >= PATIENCE_MAX:
                print(f"Early stopping at epoch {ep} (best @ {best_epoch}, val={best_val:.4f})")
                break

    print(f"Best val MSE (normalised space): {best_val:.5f} @ epoch {best_epoch}")

    # Predict test with best model
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state"])
    model.eval()
    preds = []
    with torch.no_grad():
        for sx, _ty, s_id, t_id in test_loader:
            sx = sx.to(DEVICE)
            s_id = s_id.to(DEVICE)
            t_id = t_id.to(DEVICE)
            pred = sx + model(sx, s_id, t_id)
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)  # (N_test, T, V) normalised
    preds = preds * std + mean  # denormalise

    # Clip to physically-plausible ranges (derived from train data)
    limits = {
        "temp": (-40.0, 55.0),
        "dewpoint": (-45.0, 35.0),
        "wind_speed": (0.0, 35.0),
    }
    for vi, v in enumerate(VARS):
        lo, hi = limits[v]
        preds[..., vi] = np.clip(preds[..., vi], lo, hi)

    return preds, stations


def write_submission(test_df: pd.DataFrame, preds: np.ndarray, out_path: str):
    out = {"id": test_df["id"].values}
    # Column order per challenge example: target_temp_i, target_dewpoint_i, target_wind_speed_i ... interleaved by hour
    cols = ["id"]
    for i in range(T):
        for vi, v in enumerate(VARS):
            cols.append(f"target_{v}_{i}")
            out[f"target_{v}_{i}"] = preds[:, i, vi]
    sub = pd.DataFrame(out, columns=cols)
    sub.to_csv(out_path, index=False)
    print(f"Wrote submission: {out_path}  shape={sub.shape}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    print(f"Train: {train_df.shape}  Test: {test_df.shape}")

    model_path = os.path.join(OUT_DIR, "model.pt")
    preds, _stations = train_model(train_df, test_df, model_path)
    write_submission(test_df, preds, OUT_PATH)


if __name__ == "__main__":
    main()
