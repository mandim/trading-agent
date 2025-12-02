# preprocess_data.py
import numpy as np, pandas as pd, hashlib, json, os
from numba import njit

def cache_key(meta: dict) -> str:
    blob = json.dumps(meta, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]

@njit
def ema(arr, alpha):
    out = np.empty_like(arr)
    s = 0.0
    started = False
    for i in range(arr.size):
        x = arr[i]
        if not started:
            s = x; started = True
        else:
            s = alpha * x + (1 - alpha) * s
        out[i] = s
    return out

@njit
def rsi(prices, window=14):
    gains = np.zeros(prices.size, np.float32)
    losses = np.zeros(prices.size, np.float32)
    for i in range(1, prices.size):
        d = prices[i] - prices[i-1]
        if d >= 0:
            gains[i] = d
        else:
            losses[i] = -d
    alpha = 1.0 / window
    avg_gain = ema(gains, alpha)
    avg_loss = ema(losses, alpha)
    rs = avg_gain / (avg_loss + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out

@njit
def atr(high, low, close, window=14):
    n = close.size
    tr = np.empty(n, np.float32)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    alpha = 1.0 / window
    return ema(tr, alpha)

def preprocess(
    bars_csv,
    ticks_csv,
    out_dir,
    ema_fast=12,
    ema_slow=26,
    rsi_w=14,
    atr_w=14,
    train_fraction=0.7,
):
    os.makedirs(out_dir, exist_ok=True)

    # Load bars
    bars = pd.read_csv(
        bars_csv,
        usecols=["Date", "Open", "High", "Low", "Close", "Volume"],
        dtype={
            "Open": "float32",
            "High": "float32",
            "Low": "float32",
            "Close": "float32",
            "Volume": "float32",
        },
    )

    # Convert date to epoch seconds
    bars["Date"] = pd.to_datetime(
        bars["Date"],
        format="%Y-%m-%d",
        utc=False,
        exact=True,
        errors="raise",
    ).astype("int64") // 10**9

    bt = bars["Date"].to_numpy(np.int64)
    o = bars["Open"].to_numpy(np.float32)
    h = bars["High"].to_numpy(np.float32)
    l = bars["Low"].to_numpy(np.float32)
    c = bars["Close"].to_numpy(np.float32)

    # --- Indicators ---
    alpha_f = 2.0 / (ema_fast + 1)
    alpha_s = 2.0 / (ema_slow + 1)
    ema_f = ema(c, alpha_f).astype(np.float32)
    ema_s = ema(c, alpha_s).astype(np.float32)
    macd = (ema_f - ema_s).astype(np.float32)
    macd_sig = ema(macd, 2.0 / (9 + 1)).astype(np.float32)
    _rsi = rsi(c, rsi_w).astype(np.float32)
    _atr = atr(h, l, c, atr_w).astype(np.float32)

    # --- Multi-scale returns ---
    ret_1 = np.concatenate([[0.0], np.diff(c) / c[:-1]]).astype(np.float32)
    ret_5 = (c / np.roll(c, 5) - 1.0).astype(np.float32)
    ret_20 = (c / np.roll(c, 20) - 1.0).astype(np.float32)
    ret_5[:5] = 0.0
    ret_20[:20] = 0.0

    # --- Regime ---
    above_ema_slow = (c > ema_s).astype(np.float32)
    ema_dist = ((c - ema_s) / ema_s).astype(np.float32)
    price_range = ((h - l) / c).astype(np.float32)

    # --- Volatility ---
    ret_std_20 = pd.Series(ret_1).rolling(20, min_periods=1).std().to_numpy(np.float32)
    ret_std_50 = pd.Series(ret_1).rolling(50, min_periods=1).std().to_numpy(np.float32)

    # --- Combine ---
    bars_features = np.column_stack([
        c, ema_f, ema_s, macd, macd_sig, _rsi, _atr,
        ret_1, ret_5, ret_20,
        above_ema_slow, ema_dist,
        price_range, ret_std_20, ret_std_50,
    ]).astype(np.float32)

    # --- Compute and save scaler for bar features (based on train split) ---
    n_bars = bars_features.shape[0]
    split_idx = int(n_bars * float(train_fraction))
    if split_idx <= 0:
        raise ValueError("train_fraction results in empty training set for scaler.")

    train_feats = bars_features[:split_idx]
    bars_mean = train_feats.mean(axis=0)
    bars_std = train_feats.std(axis=0) + 1e-8

    np.save(os.path.join(out_dir, "bars_mean.npy"), bars_mean.astype(np.float32))
    np.save(os.path.join(out_dir, "bars_std.npy"), bars_std.astype(np.float32))

    # --- Ticks ---
    ticks = pd.read_csv(
        ticks_csv,
        usecols=["Timestamp", "Ask price", "Bid price"],
        dtype={"Ask price": "float32", "Bid price": "float32"},
    )

    ticks["Timestamp"] = ticks["Timestamp"].str.replace(
        r'(\d{2}:\d{2}:\d{2}):(\d+)', r'\1.\2',
        regex=True,
    )
    ticks["Timestamp"] = pd.to_datetime(
        ticks["Timestamp"],
        format="%Y%m%d %H:%M:%S.%f",
        utc=False,
        errors="raise",
    ).astype("int64") // 10**9

    tt = ticks["Timestamp"].to_numpy(np.int64)
    ask = ticks["Ask price"].to_numpy(np.float32)
    bid = ticks["Bid price"].to_numpy(np.float32)

    # Tick → bar alignment
    bar_start = bt
    bar_end = np.empty_like(bar_start)
    bar_end[:-1] = bar_start[1:]
    bar_end[-1] = np.iinfo(np.int64).max

    tick_to_bar = np.empty(tt.size, dtype=np.int32)
    i = 0  # tick idx
    j = 0  # bar idx
    while i < tt.size and j < bar_start.size:
        t = tt[i]
        while j + 1 < bar_start.size and t >= bar_end[j]:
            j += 1
        if (t >= bar_start[j]) and (t < bar_end[j]):
            tick_to_bar[i] = j
            i += 1
        else:
            i += 1

    # Save artifacts
    np.save(os.path.join(out_dir, "bars_features.npy"), bars_features)
    np.save(os.path.join(out_dir, "bar_times.npy"), bt)
    np.save(os.path.join(out_dir, "tick_ask.npy"), ask)
    np.save(os.path.join(out_dir, "tick_bid.npy"), bid)
    np.save(os.path.join(out_dir, "tick_to_bar.npy"), tick_to_bar)

    meta = {
        "bars_csv": os.path.abspath(bars_csv),
        "ticks_csv": os.path.abspath(ticks_csv),
        "bars_mtime": os.path.getmtime(bars_csv),
        "ticks_mtime": os.path.getmtime(ticks_csv),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_w": rsi_w,
        "atr_w": atr_w,
        "train_fraction": float(train_fraction),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, sort_keys=True)

if __name__ == "__main__":
    # Example:
    preprocess("data/EURUSD_Daily.csv", "data/EURUSD_Ticks.csv", "cache_fx_EURUSD_D1")
