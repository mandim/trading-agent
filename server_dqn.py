import os, json, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import zmq

# ---------- Indicator helpers (mirror preprocessing.py) ----------

def ema_np(arr, alpha: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    out = np.empty_like(arr, dtype=np.float32)
    s = 0.0
    started = False
    for i in range(arr.size):
        x = float(arr[i])
        if not started:
            s = x
            started = True
        else:
            s = alpha * x + (1.0 - alpha) * s
        out[i] = s
    return out


def rsi_np(prices: np.ndarray, window: int = 14) -> np.ndarray:
    prices = np.asarray(prices, dtype=np.float32)
    n = prices.size
    gains = np.zeros(n, dtype=np.float32)
    losses = np.zeros(n, dtype=np.float32)

    for i in range(1, n):
        d = prices[i] - prices[i - 1]
        if d >= 0:
            gains[i] = d
        else:
            losses[i] = -d

    alpha = 1.0 / float(window)
    avg_gain = ema_np(gains, alpha)
    avg_loss = ema_np(losses, alpha)

    rs = avg_gain / (avg_loss + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.astype(np.float32)


def atr_np(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    high = np.asarray(high, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    close = np.asarray(close, dtype=np.float32)
    n = close.size

    tr = np.empty(n, dtype=np.float32)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    alpha = 1.0 / float(window)
    return ema_np(tr, alpha).astype(np.float32)


def compute_bar_features_from_raw(
    bars_raw: np.ndarray,
    ema_fast: int = 12,
    ema_slow: int = 26,
    rsi_w: int = 14,
    atr_w: int = 14,
) -> np.ndarray:
    """
    EXACT feature layout used in preprocessing.py (15 feats/bar):

      [c, ema_f, ema_s, macd, macd_sig, rsi, atr,
       ret_1, ret_5, ret_20,
       above_ema_slow, ema_dist,
       price_range, ret_std_20, ret_std_50]

    bars_raw shape (L, >=4): 0=Open, 1=High, 2=Low, 3=Close, [4=Volume ignored]
    """
    bars_raw = np.asarray(bars_raw, dtype=np.float32)
    if bars_raw.ndim != 2 or bars_raw.shape[1] < 4:
        raise ValueError(f"bars_raw must be 2D with at least 4 cols (O,H,L,C), got {bars_raw.shape}")

    h = bars_raw[:, 1]
    l = bars_raw[:, 2]
    c = bars_raw[:, 3]
    n = c.size

    alpha_f = 2.0 / (ema_fast + 1.0)
    alpha_s = 2.0 / (ema_slow + 1.0)
    ema_f = ema_np(c, alpha_f)
    ema_s = ema_np(c, alpha_s)
    macd = (ema_f - ema_s).astype(np.float32)
    macd_sig = ema_np(macd, 2.0 / (9.0 + 1.0))
    rsi_vals = rsi_np(c, rsi_w)
    atr_vals = atr_np(h, l, c, atr_w)

    # returns (match preprocessing)
    ret_1 = np.zeros(n, dtype=np.float32)
    if n > 1:
        ret_1[1:] = (c[1:] - c[:-1]) / np.maximum(1e-12, c[:-1])

    ret_5 = np.zeros(n, dtype=np.float32)
    ret_20 = np.zeros(n, dtype=np.float32)
    if n > 5:
        ret_5[5:] = (c[5:] / np.maximum(1e-12, c[:-5])) - 1.0
    if n > 20:
        ret_20[20:] = (c[20:] / np.maximum(1e-12, c[:-20])) - 1.0

    above_ema_slow = (c > ema_s).astype(np.float32)
    ema_dist = ((c - ema_s) / np.maximum(1e-12, ema_s)).astype(np.float32)
    price_range = ((h - l) / np.maximum(1e-12, c)).astype(np.float32)

    # Rolling std (pandas rolling std uses ddof=1)
    def rolling_std_ddof1(x: np.ndarray, window: int) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        for i in range(x.size):
            start = max(0, i - window + 1)
            w = x[start:i+1].astype(np.float64)
            if w.size < 2:
                out[i] = np.nan
            else:
                m = w.mean()
                var = ((w - m) ** 2).sum() / (w.size - 1)
                out[i] = np.float32(np.sqrt(max(0.0, var)))
        return out

    ret_std_20 = rolling_std_ddof1(ret_1, 20)
    ret_std_50 = rolling_std_ddof1(ret_1, 50)

    bars_features = np.column_stack([
        c, ema_f, ema_s, macd, macd_sig, rsi_vals, atr_vals,
        ret_1, ret_5, ret_20,
        above_ema_slow, ema_dist,
        price_range, ret_std_20, ret_std_50
    ]).astype(np.float32)

    return bars_features


# ==== Match the training net (train_dqn.py) ====
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


ACTIONS = {0: "HOLD", 1: "BUY", 2: "SELL"}

class RunningNorm:
    """Expanding mean/std using Welford (matches env 'history up to now' much closer than rolling)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        x = float(x)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, xs):
        arr = np.asarray(xs, dtype=np.float64).ravel()
        if arr.size == 0:
            return
        for x in arr:
            self.update(float(x))

    def stats(self):
        if self.n < 2:
            return self.mean, 1e-8
        var = self.M2 / self.n  # ddof=0 like np.std default
        return self.mean, (var ** 0.5) + 1e-8

    def normalize(self, x: float):
        m, s = self.stats()
        return float((float(x) - m) / s)

def load_bars_scaler(cache_dir: str, normalize_bars: bool):
    mean_p = os.path.join(cache_dir, "bars_mean.npy")
    std_p = os.path.join(cache_dir, "bars_std.npy")
    if normalize_bars and os.path.exists(mean_p) and os.path.exists(std_p):
        bars_mean = np.load(mean_p).astype(np.float32)
        bars_std = np.load(std_p).astype(np.float32)
        return bars_mean, bars_std
    return None, None

def load_price_norm_cache(cache_dir: str):
    """
    Load tick data and precompute cumulative stats so we can match TradingEnv's
    full-history price normalization at bar-open ticks.
    """
    try:
        tick_ask = np.load(os.path.join(cache_dir, "tick_ask.npy"), mmap_mode="r")
        tick_bid = np.load(os.path.join(cache_dir, "tick_bid.npy"), mmap_mode="r")
        tick_to_bar = np.load(os.path.join(cache_dir, "tick_to_bar.npy"), mmap_mode="r")
        bar_times = np.load(os.path.join(cache_dir, "bar_times.npy"), mmap_mode="r")
    except Exception:
        return None

    tick_ask = np.asarray(tick_ask, dtype=np.float64)
    tick_bid = np.asarray(tick_bid, dtype=np.float64)
    tick_to_bar = np.asarray(tick_to_bar, dtype=np.int64)
    bar_times = np.asarray(bar_times, dtype=np.int64)

    spread = tick_ask - tick_bid

    def cumsum_and_sq(x: np.ndarray):
        return np.cumsum(x), np.cumsum(x * x)

    ask_sum, ask_sum_sq = cumsum_and_sq(tick_ask)
    bid_sum, bid_sum_sq = cumsum_and_sq(tick_bid)
    spr_sum, spr_sum_sq = cumsum_and_sq(spread)

    # Map each bar to its first tick index (bar-open tick)
    bar_start_tick = np.searchsorted(tick_to_bar, np.arange(bar_times.size), side="left")

    return {
        "bar_times": bar_times,
        "bar_start_tick": bar_start_tick,
        "ask_sum": ask_sum,
        "ask_sum_sq": ask_sum_sq,
        "bid_sum": bid_sum,
        "bid_sum_sq": bid_sum_sq,
        "spr_sum": spr_sum,
        "spr_sum_sq": spr_sum_sq,
        "n_ticks": int(tick_ask.size),
    }

def _stats_up_to_tick(cumsum: np.ndarray, cumsum_sq: np.ndarray, idx: int):
    idx = int(idx)
    if idx < 0:
        idx = 0
    n = idx + 1
    s = float(cumsum[idx])
    ss = float(cumsum_sq[idx])
    mean = s / n
    var = max(0.0, (ss / n) - (mean * mean))
    std = (var ** 0.5) + 1e-8
    return mean, std

def get_price_stats(price_cache: dict, bar_index: int | None):
    if price_cache is None or bar_index is None:
        return None
    bar_index = int(bar_index)
    if bar_index < 0 or bar_index >= price_cache["bar_start_tick"].size:
        return None

    tick_idx = int(price_cache["bar_start_tick"][bar_index])
    if tick_idx < 0 or tick_idx >= price_cache["n_ticks"]:
        return None

    ask_mean, ask_std = _stats_up_to_tick(price_cache["ask_sum"], price_cache["ask_sum_sq"], tick_idx)
    bid_mean, bid_std = _stats_up_to_tick(price_cache["bid_sum"], price_cache["bid_sum_sq"], tick_idx)
    spr_mean, spr_std = _stats_up_to_tick(price_cache["spr_sum"], price_cache["spr_sum_sq"], tick_idx)

    return {
        "ask_mean": ask_mean,
        "ask_std": ask_std,
        "bid_mean": bid_mean,
        "bid_std": bid_std,
        "spr_mean": spr_mean,
        "spr_std": spr_std,
    }

def _parse_bar_time(bar_time_str: str) -> int | None:
    s = str(bar_time_str).strip()
    if not s:
        return None
    s = s.replace("T", " ")
    if "." in s[:10]:
        s = s.replace(".", "-")
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return int(dt.timestamp())
        except ValueError:
            continue
    return None


def build_obs(
    bar_window: np.ndarray,
    bars_mean, bars_std,
    ask: float, bid: float,
    price_norms,
    position_side: int,
    entry_ask,
    entry_bid,
    pos_age_bars: float,
    pip_decimal: float,
    sl_pips: float,
    lot: float,
    exchange_rate: float,
    update_norms: bool = True,
    price_stats: dict | None = None,
):
    """
    Build obs aligned with TradingEnv._get_observation():
      [window_len * bar_features] + [ask_norm, bid_norm, spread_norm] + [pos_dir, pos_pnl_R, pos_age]
    """
    if bars_mean is not None and bars_std is not None:
        bar_window = (bar_window - bars_mean) / (bars_std + 1e-8)

    ask = float(ask)
    bid = float(bid)
    spread = ask - bid

    if price_stats is not None:
        a = (ask - float(price_stats["ask_mean"])) / float(price_stats["ask_std"])
        b = (bid - float(price_stats["bid_mean"])) / float(price_stats["bid_std"])
        s = (spread - float(price_stats["spr_mean"])) / float(price_stats["spr_std"])
    else:
        if update_norms:
            price_norms["ask"].update(ask)
            price_norms["bid"].update(bid)
            price_norms["spread"].update(spread)

        a = price_norms["ask"].normalize(ask)
        b = price_norms["bid"].normalize(bid)
        s = price_norms["spread"].normalize(spread)

    pos_dir = 0.0
    pos_pnl_R = 0.0
    pos_age = float(pos_age_bars) / 1000.0

    if int(position_side) != 0 and entry_ask is not None and entry_bid is not None:
        pos_dir = 1.0 if int(position_side) > 0 else -1.0

        # pip value in USD (same as your current)
        pip_value_usd = (100000.0 * float(pip_decimal)) / max(1e-12, float(exchange_rate))
        pip_value_usd *= float(lot)

        # Match env:
        # long pnl_pips  = (bid - entry_ask) / pip_decimal
        # short pnl_pips = (entry_bid - ask) / pip_decimal
        if pos_dir > 0:
            pnl_pips = (float(bid) - float(entry_ask)) / float(pip_decimal)
        else:
            pnl_pips = (float(entry_bid) - float(ask)) / float(pip_decimal)

        float_profit_usd = float(pnl_pips) * float(pip_value_usd)
        one_R_usd = max(1e-9, float(pip_value_usd) * float(sl_pips))
        pos_pnl_R = float(float_profit_usd / one_R_usd)


    pos_feats = np.array([pos_dir, pos_pnl_R, pos_age], dtype=np.float32)

    obs = np.hstack([
        bar_window.ravel(),
        np.array([a, b, s], dtype=np.float32),
        pos_feats
    ]).astype(np.float32)

    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    return obs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache_fx_EURUSD_D1_fx")
    parser.add_argument("--model_path", default="models/dqn_best.pt")
    parser.add_argument("--bind", default="tcp://127.0.0.1:6000")
    parser.add_argument("--window_len", type=int, default=32)
    parser.add_argument("--normalize_bars", type=int, default=1)
    parser.add_argument("--price_norm_lookback", type=int, default=20000)
    parser.add_argument("--log_every", type=int, default=50, help="print debug once every N steps (avoid slowing tester)")
    parser.add_argument("--default_sl_pips", type=float, default=40.0, help="must match training sl_pips if EA doesn't send")
    args = parser.parse_args()

    bars_mean, bars_std = load_bars_scaler(args.cache_dir, bool(args.normalize_bars))
    price_cache = load_price_norm_cache(args.cache_dir)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[serve] Listening on {args.bind}")

    # Rolling price normalization
    price_norms = {
        "ask": RunningNorm(),
        "bid": RunningNorm(),
        "spread": RunningNorm(),
    }

    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    n_feats = None
    step_count = 0
    action_hist = {0: 0, 1: 0, 2: 0}
    norm_diag = {"price_stats_ok": 0, "price_stats_missing": 0}

    # Feature-history minimum to compute std_50 properly
    min_raw_len = max(args.window_len, 60)

    while True:
        try:
            msg = sock.recv()
        except KeyboardInterrupt:
            print("[serve] KeyboardInterrupt, shutting down...")
            break

        try:
            req = json.loads(msg.decode("utf-8"))
        except Exception:
            sock.send_json({"ok": False, "error": "invalid_json"})
            continue

        cmd = req.get("cmd", "")
        if cmd == "shutdown":
            sock.send_json({"ok": True, "reply": "shutting_down"})
            print("[serve] Received shutdown command.")
            break

        if cmd == "reset":
            price_norms = {
                "ask": RunningNorm(),
                "bid": RunningNorm(),
                "spread": RunningNorm(),
            }
            sock.send_json({"ok": True, "reply": "reset_done"})
            continue

        if cmd != "step":
            sock.send_json({"ok": False, "error": "unknown_cmd"})
            continue

        ask = req.get("ask")
        bid = req.get("bid")
        if ask is None or bid is None:
            sock.send_json({"ok": False, "error": "missing_fields_ask_bid"})
            continue

        bar_window_raw = req.get("bar_window_raw")
        bar_window = req.get("bar_window")  # legacy

        if bar_window_raw is not None:
            bars_raw = np.asarray(bar_window_raw, dtype=np.float32)

            if bars_raw.ndim != 2 or bars_raw.shape[0] < min_raw_len:
                sock.send_json({"ok": False, "error": f"bad_bar_window_raw_shape_need_len_ge_{min_raw_len}"})
                continue

            try:
                bar_feats_full = compute_bar_features_from_raw(bars_raw)
                bar_window = bar_feats_full[-args.window_len:, :]
            except Exception as e:
                sock.send_json({"ok": False, "error": f"feature_computation_failed: {str(e)}"})
                continue

        elif bar_window is not None:
            bar_window = np.asarray(bar_window, dtype=np.float32)
        else:
            sock.send_json({"ok": False, "error": "missing_bar_window_or_raw"})
            continue

        if bar_window.ndim != 2 or bar_window.shape[0] != args.window_len:
            sock.send_json({"ok": False, "error": "bad_bar_window_shape"})
            continue

        if n_feats is None:
            n_feats = int(bar_window.shape[1])
            obs_dim = args.window_len * n_feats + 6
            model = QNet(obs_dim, 3).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            print(f"[serve] Model ready: obs_dim={obs_dim}, n_feats={n_feats}")

        # Tick-based normalization update (if provided by EA)
        tick_asks = req.get("tick_asks")
        tick_bids = req.get("tick_bids")
        use_tick_norms = False
        if tick_asks is not None and tick_bids is not None:
            try:
                tick_asks_np = np.asarray(tick_asks, dtype=np.float64)
                tick_bids_np = np.asarray(tick_bids, dtype=np.float64)
                if tick_asks_np.size != 0 and tick_asks_np.size == tick_bids_np.size:
                    price_norms["ask"].update_batch(tick_asks_np)
                    price_norms["bid"].update_batch(tick_bids_np)
                    price_norms["spread"].update_batch(tick_asks_np - tick_bids_np)
                    use_tick_norms = True
            except Exception:
                pass

        bar_index = req.get("bar_index", None)
        if bar_index is None:
            bar_time = req.get("bar_time", None)
            if bar_time is not None and price_cache is not None:
                epoch = _parse_bar_time(bar_time)
                if epoch is not None:
                    bt = price_cache["bar_times"]
                    idx = int(np.searchsorted(bt, epoch, side="left"))
                    if idx < bt.size and int(bt[idx]) == int(epoch):
                        bar_index = idx

        price_stats = get_price_stats(price_cache, bar_index) if price_cache is not None else None
        if price_stats is None:
            norm_diag["price_stats_missing"] += 1
        else:
            norm_diag["price_stats_ok"] += 1

        sl_pips = req.get("sl_pips", None)
        if sl_pips is None:
            sl_pips = args.default_sl_pips

        obs = build_obs(
            bar_window, bars_mean, bars_std,
            float(ask), float(bid), price_norms,
            position_side=int(req.get("position_side", 0)),
            entry_ask=req.get("entry_ask", None),
            entry_bid=req.get("entry_bid", None),
            pos_age_bars=float(req.get("pos_age_bars", 0.0)),
            pip_decimal=float(req.get("pip_decimal", 0.0001)),
            sl_pips=float(sl_pips),
            lot=float(req.get("lot", 1.0)),
            exchange_rate=float(req.get("exchange_rate", 1.0)),
            update_norms=not use_tick_norms,
            price_stats=price_stats,
        )

        q = model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        a = int(q.argmax(dim=1).item())

        action_hist[a] += 1
        step_count += 1

        if args.log_every > 0 and (step_count % args.log_every == 0):
            print("[serve] obs stats:",
            "bars=", float(np.mean(obs[:args.window_len*n_feats])),
            "price=", obs[args.window_len*n_feats:args.window_len*n_feats+3].tolist(),
            "pos=", obs[-3:].tolist())
            total = max(1, norm_diag["price_stats_ok"] + norm_diag["price_stats_missing"])
            miss_pct = 100.0 * norm_diag["price_stats_missing"] / total
            print(
                "[serve] norm diag:",
                "price_stats_ok=", norm_diag["price_stats_ok"],
                "price_stats_missing=", norm_diag["price_stats_missing"],
                f"missing_pct={miss_pct:.2f}%"
            )

        out = {
            "ok": True,
            "a": a,
            "action": ACTIONS.get(a, "HOLD"),
            "q": q.squeeze(0).detach().cpu().tolist(),
        }
        sock.send_json(out)

    sock.close(0)
    ctx.term()
    print("[serve] Closed socket and context. Bye.")


if __name__ == "__main__":
    main()
