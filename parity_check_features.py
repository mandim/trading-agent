import numpy as np
import pandas as pd

from preprocessing import ema, rsi, atr
from server_dqn import compute_bar_features_from_raw


def compute_features_preprocessing(
    bars_raw: np.ndarray,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_w: int = 14,
    atr_w: int = 14,
) -> np.ndarray:
    bars_raw = np.asarray(bars_raw, dtype=np.float32)
    if bars_raw.ndim != 2 or bars_raw.shape[1] < 4:
        raise ValueError(f"bars_raw must be 2D with at least 4 cols (O,H,L,C), got {bars_raw.shape}")

    h = bars_raw[:, 1]
    l = bars_raw[:, 2]
    c = bars_raw[:, 3]

    alpha_f = 2.0 / (ema_fast + 1)
    alpha_s = 2.0 / (ema_slow + 1)
    ema_f = ema(c, alpha_f).astype(np.float32)
    ema_s = ema(c, alpha_s).astype(np.float32)
    macd = (ema_f - ema_s).astype(np.float32)
    macd_sig = ema(macd, 2.0 / (9 + 1)).astype(np.float32)
    _rsi = rsi(c, rsi_w).astype(np.float32)
    _atr = atr(h, l, c, atr_w).astype(np.float32)

    ret_1 = np.concatenate([[0.0], np.diff(c) / c[:-1]]).astype(np.float32)
    ret_5 = (c / np.roll(c, 5) - 1.0).astype(np.float32)
    ret_20 = (c / np.roll(c, 20) - 1.0).astype(np.float32)
    ret_5[:5] = 0.0
    ret_20[:20] = 0.0

    above_ema_slow = (c > ema_s).astype(np.float32)
    ema_dist = ((c - ema_s) / ema_s).astype(np.float32)
    price_range = ((h - l) / c).astype(np.float32)

    ret_std_20 = pd.Series(ret_1).rolling(20, min_periods=1).std().to_numpy(np.float32)
    ret_std_50 = pd.Series(ret_1).rolling(50, min_periods=1).std().to_numpy(np.float32)

    bars_features = np.column_stack([
        c, ema_f, ema_s, macd, macd_sig, _rsi, _atr,
        ret_1, ret_5, ret_20,
        above_ema_slow, ema_dist,
        price_range, ret_std_20, ret_std_50,
    ]).astype(np.float32)
    return bars_features


def main():
    rng = np.random.default_rng(42)
    n = 80
    c = np.cumsum(rng.normal(0, 0.5, size=n)).astype(np.float32) + 100.0
    o = c + rng.normal(0, 0.1, size=n).astype(np.float32)
    h = np.maximum(o, c) + rng.uniform(0.0, 0.2, size=n).astype(np.float32)
    l = np.minimum(o, c) - rng.uniform(0.0, 0.2, size=n).astype(np.float32)
    v = rng.integers(100, 1000, size=n).astype(np.float32)

    bars_raw = np.column_stack([o, h, l, c, v]).astype(np.float32)

    feats_pre = compute_features_preprocessing(bars_raw)
    feats_srv = compute_bar_features_from_raw(bars_raw)

    if feats_pre.shape != feats_srv.shape:
        raise SystemExit(f"Shape mismatch: preprocessing={feats_pre.shape}, server={feats_srv.shape}")

    diff = np.abs(feats_pre - feats_srv)
    max_diff = float(np.nanmax(diff))
    mean_diff = float(np.nanmean(diff))
    ok = np.allclose(feats_pre, feats_srv, rtol=1e-4, atol=1e-5, equal_nan=True)

    print("feature_parity_ok=", bool(ok))
    print("max_abs_diff=", max_diff)
    print("mean_abs_diff=", mean_diff)


if __name__ == "__main__":
    main()
