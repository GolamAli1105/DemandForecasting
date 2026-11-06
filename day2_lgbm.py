import pandas as pd, numpy as np, lightgbm as lgb

H = 28  # forecast horizon

# ---------- 1. Load ----------
sales  = pd.read_csv("data/sales_train_evaluation.csv")
cal    = pd.read_csv("data/calendar.csv")
prices = pd.read_csv("data/sell_prices.csv")

sub = sales[(sales.store_id == "CA_1") & (sales.cat_id == "FOODS")].reset_index(drop=True)
sub["sid"] = np.arange(len(sub))                    # series id 0..1436
day_cols = [c for c in sub.columns if c.startswith("d_")]
n_series, T = len(sub), len(day_cols)               # 1437, 1941
Y = sub[day_cols].to_numpy(dtype=np.float32)        # (series, days), used for baselines

# ---------- 2. Wide -> long (one row per product per day) ----------
df = sub[["sid", "item_id", "dept_id"] + day_cols].melt(
    id_vars=["sid", "item_id", "dept_id"], var_name="d", value_name="sales")
df["d"] = df["d"].str[2:].astype(int)               # "d_17" -> 17

# ---------- 3. Calendar + price features (known in advance, so no leakage) ----------
cal["d"] = cal["d"].str[2:].astype(int)
cal["is_event"] = cal["event_name_1"].notna().astype(int)
df = df.merge(cal[["d", "wm_yr_wk", "wday", "month", "snap_CA", "is_event"]], on="d", how="left")

p = prices[prices.store_id == "CA_1"][["item_id", "wm_yr_wk", "sell_price"]]
df = df.merge(p, on=["item_id", "wm_yr_wk"], how="left")   # NaN = not on sale yet

df = df.sort_values(["sid", "d"]).reset_index(drop=True)   # order matters for shift/rolling
df["price_rel"] = df["sell_price"] / df.groupby("sid")["sell_price"].transform(
    lambda s: s.shift(H).expanding().mean())
df["dept_code"] = df["dept_id"].astype("category").cat.codes

# ---------- 4. Lag + rolling features (ALL at least H=28 days back) ----------
g = df.groupby("sid")["sales"]
for lag in [28, 35, 42, 49, 56]:
    df[f"lag_{lag}"] = g.shift(lag)

base = g.shift(H)                                   # sales as known 28 days earlier
for w in [7, 28]:
    df[f"rmean_{w}"] = base.groupby(df["sid"]).transform(lambda s: s.rolling(w).mean())
df["rstd_28"] = base.groupby(df["sid"]).transform(lambda s: s.rolling(28).std())

FEATS = ["lag_28", "lag_35", "lag_42", "lag_49", "lag_56",
         "rmean_7", "rmean_28", "rstd_28",
         "sell_price", "price_rel",
         "wday", "month", "snap_CA", "is_event", "dept_code"]

# ---------- 5. Metric ----------
def mase(train, test, pred, m=7):
    scale = np.abs(train[:, m:] - train[:, :-m]).mean(axis=1)
    err = np.abs(test - pred).mean(axis=1)
    ok = scale > 0
    return (err[ok] / scale[ok]).mean()

# ---------- 6. Rolling-origin validation: 3 folds ----------
results = []
for cutoff in [T - 3 * H, T - 2 * H, T - H]:        # last known day for each fold
    tr = df[(df.d > 100) & (df.d <= cutoff)]        # skip warm-up days where lags are NaN
    te = df[(df.d > cutoff) & (df.d <= cutoff + H)]
    assert len(te) == n_series * H

    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.1,   # good for count data with many zeros
        n_estimators=500, learning_rate=0.05, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        min_child_samples=50, verbose=-1)
    model.fit(tr[FEATS], tr["sales"])

    pred = np.clip(model.predict(te[FEATS]), 0, None).reshape(n_series, H)

    train_arr, test_arr = Y[:, :cutoff], Y[:, cutoff:cutoff + H]
    snaive = np.tile(train_arr[:, -7:], (1, H // 7))
    m_sn, m_lgb = mase(train_arr, test_arr, snaive), mase(train_arr, test_arr, pred)
    results.append((m_sn, m_lgb))
    print(f"cutoff d_{cutoff}: sNaive {m_sn:.3f} | LightGBM {m_lgb:.3f}")

res = np.array(results)
print(f"\nAVG  sNaive {res[:,0].mean():.3f} | LightGBM {res[:,1].mean():.3f}")
print("Feature importance:")
print(pd.Series(model.feature_importances_, FEATS).sort_values(ascending=False))