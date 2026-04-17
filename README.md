# Weather Translation Challenge

Shipd "Weather Translation" challenge: given 3 days of hourly weather observations
(temperature, dewpoint, wind speed) from a *source* station, predict the
corresponding 3-day series for a *target* station during the same time window.

- **Domain:** sequence-to-sequence
- **Score:** `max(0, 1 - nRMSE)` pooled over all 216 target values per sample
  (3 variables × 72 hours) across all test samples. Reference mean-prediction
  baseline ≈ 0.17.
- **Rules:** train from scratch, no pretrained weather models, no external
  climate data.

## Data

- `public/train.csv` — 12,444 paired source/target windows across 12 station
  pairs (1,037 per pair).
- `public/test.csv` — 3,640 source windows across 14 station pairs. **Two of
  those 14 pairs never appear in training** — the model must generalise to
  `(H → F)` and `(I → A)`.
- `public/sample_submission.csv` — target schema.

9 anonymous stations `Station_A`–`Station_I`. Station identities are withheld.

## Approach (current `solution.py` = v5)

Sequence-to-sequence BiLSTM with per-station embeddings, operating in
**climate-anomaly space**:

1. Per-station climate mean is computed from every observation of that station
   in the training set (source or target role, pooled).
2. Source series → subtract source station's climate mean → normalise by the
   global anomaly std.
3. BiLSTM encoder with source/target station embeddings + sin/cos time
   features predicts the residual from identity.
4. Add back the target station's climate mean and denormalise to get the
   target forecast.

This factorises prediction into (a) an exact per-station climate offset
handled in closed form and (b) a learned station-pair anomaly transformation,
which is the piece that actually has to generalise across unseen pairs.

A simple uniform-weight ensemble over multiple LSTM seeds (deterministic but
diverse initialisations) is averaged in normalised-anomaly space before
denormalisation.

## Iteration history (`blue/vN/`)

Each folder snapshots a working `solution.py` + its `submission.csv`:

| Version | Description                                                                  | Val score | Shipd test (upload) |
|:-------:|:-----------------------------------------------------------------------------|:---------:|:-------------------:|
| v1      | Single 4-layer Transformer with separate src/tgt station embeddings          | 0.706     | 0.6032              |
| v2      | Variance-weighted loss + EMA (worse — kept for reference)                    | 0.663     | -                   |
| v3      | 5-seed Transformer ensemble                                                  | 0.741     | -                   |
| v4      | Heterogeneous ensemble (5 Transformer + 3 BiLSTM)                            | 0.790     | 0.6765              |
| v5      | Climate-anomaly BiLSTM ensemble (6 LSTM seeds, no Transformer, Colab A100)   | 0.8355    | 0.7281              |
| v6      | v5 + emb-dropout + mixup + 2 TCN seeds + 4x TTA                              | 0.8153    | 0.7092              |
| v7      | 5 LSTM + 1 GRU + 2 coupled-pair-dropout + 2 pair-agnostic (auto-pick `diverse`) | 0.8411 | *pending*           |
| v8      | Bigger BiLSTM (d_model 224, station_dim 64) + RH/VPD physical features + TTA, 7-model ensemble | **0.8499** | *pending* |

### v8 breakdown (Colab, 7 models: 5 BiLSTM + 1 BiGRU + 1 BiLSTM pd=0.3)

```
Val MSE raw:  2.3343
Val nRMSE:    0.1501
Val score:    0.8499          (+0.009 over v7 'diverse')

Per-variable MSE:   temp 2.87    dewpoint 3.34    wind_speed 0.79
Per-seed anom-norm val MSE (all ~15% lower than v7):
  lstm 1337 pd=0.0 = 0.1040   (v7: 0.1244)
  lstm 2024 pd=0.0 = 0.1043   (v7: 0.1255)
  lstm 4242 pd=0.0 = 0.1054   (v7: 0.1250)
  lstm  777 pd=0.0 = 0.1060   (v7: 0.1256)
  lstm 31337 pd=0.0 = 0.1066  (v7: 0.1256)
  gru  2718 pd=0.0 = 0.1227   (v7: 0.1399)
  lstm 1337 pd=0.3 = 0.1194   (v7: 0.1408)
```

Projected Shipd test score (using v5's val→test gap of 0.107):
~0.743 → projected rank 2 (clears ahmed_salah7's 0.7354, still ~0.022
below jigyasu's 0.7649).

The val→test gap of ~0.11 points across v1 and v4 comes from the two
station pairs in test (`H → F`, `I → A`) that never appear in training —
they account for ~14% of test samples and our validation split, stratified
over *seen* pairs only, cannot see them. v5 is the first version that
explicitly factors out the per-station climate offset so the learned
transformation is only responsible for the anomaly-to-anomaly mapping,
which should generalise across pairs much better than a pair-embedding
lookup.

### v5 breakdown (Colab, all 6 seeds)

```
Val MSE raw:  2.8061
Val nRMSE:    0.1645
Val score:    0.8355

Per-variable MSE:   temp 3.41    dewpoint 3.96    wind_speed 1.05
Per-seed anom-norm val MSE: 1337=0.1378  2024=0.1352  4242=0.1308
                            777=0.1350  31337=0.1398  2718=0.1351
```

### v6 breakdown (Colab, 6 BiLSTM + 2 TCN seeds)

```
Val MSE raw:  3.5345
Val nRMSE:    0.1847
Val score:    0.8153

Per-variable MSE:   temp 4.32    dewpoint 4.87    wind_speed 1.41
Per-seed anom-norm val MSE:
  lstm 1337=0.1476  2024=0.1479  4242=0.1517
  lstm  777=0.1444 31337=0.1504  2718=0.1455
  tcn  1337=0.2512  4242=0.2537
```

Observations: v6's per-LSTM val regressed from ~0.135 -> ~0.148 (emb-dropout
+ mixup hurt the on-distribution val), and the two TCN seeds at ~0.25 dilute
the ensemble. Whether v6 actually beats v5 on the Shipd test set depends on
whether the added regularisation closed the unseen-pair gap more than it
hurt in-distribution accuracy — a question the leaderboard answers, not
our pair-stratified val.

The ~10-point val→test gap on v1 highlighted the two unseen station pairs as
the dominant failure mode, which is what v5's climate-anomaly decomposition
targets directly.

## Layout (Shipd submission container)

Script reads from `public/` locally (or `dataset/public/` inside the grader)
and writes to `working/submission.csv`.

## Running

```bash
python solution.py
# or a pinned previous version
python blue/v4/solution.py
```

## Colab

Open `colab_run.ipynb` — it clones this repo, sets up the sandbox layout,
runs any selected `solution.py`, and validates the submission.
