import pandas as pd, numpy as np

sales = pd.read_csv("data/sales_train_evaluation.csv")

# Subset: one store, one category (~1,400 series)
sub = sales[(sales.store_id == "CA_1") & (sales.cat_id == "FOODS")].reset_index(drop=True)
day_cols = [c for c in sub.columns if c.startswith("d_")]
Y = sub[day_cols].to_numpy(dtype=float)
print("Y shape:", Y.shape)   # expect roughly (1400, 1941)

H = 28
train, test = Y[:, :-H], Y[:, -H:]

def mase(train, test, pred, m=7):
    scale = np.abs(train[:, m:] - train[:, :-m]).mean(axis=1)
    err = np.abs(test - pred).mean(axis=1)
    ok = scale > 0
    return (err[ok] / scale[ok]).mean()

naive = np.repeat(train[:, -1:], H, axis=1)
snaive = np.tile(train[:, -7:], (1, H // 7))

print("Naive MASE:  ", mase(train, test, naive))
print("sNaive MASE: ", mase(train, test, snaive))