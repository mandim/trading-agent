import os, json, argparse, numpy as np, torch, torch.nn as nn
import zmq

# ---------- Indicator helpers (mirror preprocessing.py) ----------

def ema_np(arr, alpha: float) -> np.ndarray:
    """
    Same logic as preprocessing.ema (Numba version), but pure NumPy.
    s_0 = x_0, then s_t = alpha * x_t + (1 - alpha) * s_{t-1}.
    """
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
    """
    Same as preprocessing.rsi: build gains/losses and apply EMA with alpha = 1/window.
    """
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
    """
    Same as preprocessing.atr: true range series then EMA with alpha = 1/window.
    """
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
    Recreate EXACT bar_features layout used in preprocessing.py (15 feats/bar):

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

    # --- Indicators (same as preprocessing.py) ---
    alpha_f = 2.0 / (ema_fast + 1.0)
    alpha_s = 2.0 / (ema_slow + 1.0)
    ema_f = ema_np(c, alpha_f)
    ema_s = ema_np(c, alpha_s)
    macd = (ema_f - ema_s).astype(np.float32)
    macd_sig = ema_np(macd, 2.0 / (9.0 + 1.0))
    rsi_vals = rsi_np(c, rsi_w)
    atr_vals = atr_np(h, l, c, atr_w)

    # --- Multi-scale returns (match preprocessing behavior) ---
    ret_1 = np.zeros(n, dtype=np.float32)
    if n > 1:
        ret_1[1:] = (c[1:] - c[:-1]) / np.maximum(1e-12, c[:-1])

    ret_5 = np.zeros(n, dtype=np.float32)
    ret_20 = np.zeros(n, dtype=np.float32)
    if n > 5:
        ret_5[5:] = (c[5:] / np.maximum(1e-12, c[:-5])) - 1.0
    if n > 20:
        ret_20[20:] = (c[20:] / np.maximum(1e-12, c[:-20])) - 1.0

    # --- Regime ---
    above_ema_slow = (c > ema_s).astype(np.float32)
    ema_dist = ((c - ema_s) / np.maximum(1e-12, ema_s)).astype(np.float32)
    price_range = ((h - l) / np.maximum(1e-12, c)).astype(np.float32)

    # --- Rolling std of ret_1 (pandas rolling std uses ddof=1) ---
    def rolling_std_ddof1(x: np.ndarray, window: int) -> np.ndarray:
        out = np.empty_like(x, dtype=np.float32)
        for i in range(x.size):
            start = max(0, i - window + 1)
            w = x[start:i+1].astype(np.float64)
            if w.size < 2:
                out[i] = np.nan  # ddof=1 undefined for <2 samples (matches pandas behavior)
            else:
                m = w.mean()
                var = ((w - m) ** 2).sum() / (w.size - 1)  # ddof=1
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



# ==== Match the training net (from train_dqn.py) ====
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

class RollingNorm:
    """Rolling mean/std for prices to mirror env behavior (last N ticks)."""
    def __init__(self, maxlen=20000):
        self.maxlen = int(maxlen)
        self.buf = []
    def update(self, x: float):
        self.buf.append(float(x))
        if len(self.buf) > self.maxlen:
            self.buf.pop(0)
    def stats(self):
        if not self.buf:
            return None, None
        arr = np.asarray(self.buf, dtype=np.float64)
        m = float(arr.mean())
        s = float(arr.std() + 1e-8)
        return m, s
    def normalize(self, x: float):
        m, s = self.stats()
        if m is None or s is None or s == 0.0:
            return float(x)
        return float((x - m) / s)
    
class StaticPriceNorms:
    """
    Episode-style price normalization:
      - fit once from historical ask/bid arrays
      - then keep mean/std fixed for all ticks in the episode
    """
    def __init__(self):
        self.ask_mean = None
        self.ask_std = None
        self.bid_mean = None
        self.bid_std = None
        self.spread_mean = None
        self.spread_std = None

    def fit_from_arrays(self, ask_arr, bid_arr):
        ask_arr = np.asarray(ask_arr, dtype=np.float32)
        bid_arr = np.asarray(bid_arr, dtype=np.float32)
        spread_arr = ask_arr - bid_arr

        self.ask_mean = float(ask_arr.mean())
        self.ask_std  = float(ask_arr.std() + 1e-8)

        self.bid_mean = float(bid_arr.mean())
        self.bid_std  = float(bid_arr.std() + 1e-8)

        self.spread_mean = float(spread_arr.mean())
        self.spread_std  = float(spread_arr.std() + 1e-8)

    def normalize(self, ask, bid):
        spread = ask - bid

        def nz(x, m, s):
            if m is None or s is None or s == 0.0:
                return float(x)
            return float((x - m) / (s + 1e-8))

        a = nz(ask, self.ask_mean, self.ask_std)
        b = nz(bid, self.bid_mean, self.bid_std)
        s = nz(spread, self.spread_mean, self.spread_std)
        return a, b, s


def fit_price_norms_from_cache(cache_dir: str,
                               norms: StaticPriceNorms,
                               lookback: int = 20000):
    """
    Fit StaticPriceNorms from cached tick_ask/tick_bid.

    This approximates the env behavior which, on reset, fits mean/std
    from a large window of historical ticks behind the episode start.
    Here we just use the last `lookback` ticks from cache.
    """
    try:
        tick_ask = np.load(os.path.join(cache_dir, "tick_ask.npy"), mmap_mode="r")
        tick_bid = np.load(os.path.join(cache_dir, "tick_bid.npy"), mmap_mode="r")
    except Exception as e:
        print(f"[serve] Could not load ticks for price norms: {e}")
        return

    n = int(tick_ask.shape[0])
    if n == 0:
        print("[serve] No ticks in cache for price norms.")
        return

    start = max(0, n - int(lookback))
    ask_slice = tick_ask[start:].astype(np.float32)
    bid_slice = tick_bid[start:].astype(np.float32)

    norms.fit_from_arrays(ask_slice, bid_slice)
    print(f"[serve] Price norms fitted from cache with {ask_slice.shape[0]} ticks")


def load_bars_scaler(cache_dir: str, normalize_bars: bool):
    mean_p = os.path.join(cache_dir, "bars_mean.npy")
    std_p  = os.path.join(cache_dir, "bars_std.npy")
    if normalize_bars and os.path.exists(mean_p) and os.path.exists(std_p):
        bars_mean = np.load(mean_p).astype(np.float32)
        bars_std  = np.load(std_p).astype(np.float32)
        return bars_mean, bars_std
    return None, None

def build_obs(bar_window: np.ndarray,
              bars_mean, bars_std,
              ask: float, bid: float,
              price_norms,
              # position features (from EA)
              position_side: int,
              entry_price,
              pos_age_bars: float,
              pip_decimal: float,
              sl_pips: float,
              lot: float,
              exchange_rate: float):
    """
    Build observation vector aligned with TradingEnv._get_observation():

      obs = [window_len * bar_features] + [ask_norm, bid_norm, spread_norm]
            + [pos_dir, pos_pnl_R, pos_age]

    Notes:
      - position_side: 0 flat, +1 long, -1 short
      - entry_price: for long it's entry_ask; for short it's entry_bid (see env _pnl_from_prices)
      - pos_age is scaled similarly to env (we use bars/1000.0, not ticks/1000.0)
    """
    if bars_mean is not None and bars_std is not None:
        bar_window = (bar_window - bars_mean) / (bars_std + 1e-8)

    ask = float(ask)
    bid = float(bid)
    spread = ask - bid

    if isinstance(price_norms, dict):
        price_norms["ask"].update(ask)
        price_norms["bid"].update(bid)
        price_norms["spread"].update(spread)

        a = price_norms["ask"].normalize(ask)
        b = price_norms["bid"].normalize(bid)
        s = price_norms["spread"].normalize(spread)
    else:
        a, b, s = price_norms.normalize(ask, bid)

    pos_dir = 0.0
    pos_pnl_R = 0.0
    pos_age = float(pos_age_bars) / 1000.0

    if int(position_side) != 0 and entry_price is not None:
        pos_dir = 1.0 if int(position_side) > 0 else -1.0
        pip_value_usd = (100000.0 * float(pip_decimal)) / max(1e-12, float(exchange_rate))
        pip_value_usd *= float(lot)

        if pos_dir > 0:
            pnl_pips = (float(bid) - float(entry_price)) / float(pip_decimal)
        else:
            pnl_pips = (float(entry_price) - float(ask)) / float(pip_decimal)

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
    parser.add_argument("--cache_dir", default="cache_fx_EURUSD_D1")
    parser.add_argument("--model_path", default="models/dqn_best.pt")
    parser.add_argument("--bind", default="tcp://127.0.0.1:6000")
    parser.add_argument("--window_len", type=int, default=32)
    parser.add_argument("--normalize_bars", type=int, default=1)
    parser.add_argument("--price_norm_lookback", type=int, default=20000)
    args = parser.parse_args()

    bars_mean, bars_std = load_bars_scaler(args.cache_dir, bool(args.normalize_bars))
    n_feats = None  # we infer from first request

    # ZMQ REP socket
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[serve] Listening on {args.bind}")

    # Price normalizers (rolling, like env reset fit)
    price_norms = {
        "ask": RollingNorm(args.price_norm_lookback),
        "bid": RollingNorm(args.price_norm_lookback),
        "spread": RollingNorm(args.price_norm_lookback),
    }
    # Price normalizers (static, episode-style like env reset)
    # price_norms = StaticPriceNorms()
    # fit_price_norms_from_cache(args.cache_dir, price_norms, args.price_norm_lookback)

    # Lazy model init after we know obs_dim
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    
    action_hist = {0: 0, 1: 0, 2: 0}

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

        # ---- graceful shutdown command ----
        if req.get("cmd") == "shutdown":
            sock.send_json({"ok": True, "reply": "shutting_down"})
            print("[serve] Received shutdown command from client.")
            break
        # -----------------------------------
        
        # Protocol
        # if req.get("cmd") == "reset":
        #     price_norms = {
        #         "ask": RollingNorm(args.price_norm_lookback),
        #         "bid": RollingNorm(args.price_norm_lookback),
        #         "spread": RollingNorm(args.price_norm_lookback),
        #     }
        #     sock.send_json({"ok": True, "reply": "reset_done"})
        #     continue
        if req.get("cmd") == "reset":
            price_norms = {
                "ask": RollingNorm(args.price_norm_lookback),
                "bid": RollingNorm(args.price_norm_lookback),
                "spread": RollingNorm(args.price_norm_lookback),
            }
            sock.send_json({"ok": True, "reply": "reset_done"})
            continue

        if req.get("cmd") != "step":
            sock.send_json({"ok": False, "error": "unknown_cmd"})
            continue

        # -------- Required fields --------
        ask = req.get("ask")
        bid = req.get("bid")

        # New protocol: prefer raw OHLCV if present
        bar_window_raw = req.get("bar_window_raw")  # list[list[float]] shape (L, 4/5)
        bar_window = req.get("bar_window")          # legacy: already-computed features

        if ask is None or bid is None:
            sock.send_json({"ok": False, "error": "missing_fields_ask_bid"})
            continue

        # Build feature window
        if bar_window_raw is not None:
            bars_raw = np.asarray(bar_window_raw, dtype=np.float32)
            if bars_raw.ndim != 2 or bars_raw.shape[0] != args.window_len:
                sock.send_json({"ok": False, "error": "bad_bar_window_raw_shape"})
                continue
            try:
                bar_window = compute_bar_features_from_raw(bars_raw)
            except Exception as e:
                sock.send_json({"ok": False, "error": f"feature_computation_failed: {str(e)}"})
                continue

        elif bar_window is not None:
            # legacy path: EA sends features directly
            bar_window = np.asarray(bar_window, dtype=np.float32)

        else:
            sock.send_json({"ok": False, "error": "missing_bar_window_or_raw"})
            continue

        # Sanity check final feature window
        if bar_window.ndim != 2 or bar_window.shape[0] != args.window_len:
            sock.send_json({"ok": False, "error": "bad_bar_window_shape"})
            continue

        if n_feats is None:
            n_feats = int(bar_window.shape[1])
            obs_dim = args.window_len * n_feats + 6

            # init model now that we know obs_dim
            model = QNet(obs_dim, 3).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            print(f"[serve] Model ready: obs_dim={obs_dim}, n_feats={n_feats}")

        obs = build_obs(bar_window, bars_mean, bars_std, float(ask), float(bid), price_norms,
                      position_side=int(req.get('position_side', 0)),
                      entry_price=req.get('entry_price', None),
                      pos_age_bars=float(req.get('pos_age_bars', 0.0)),
                      pip_decimal=float(req.get('pip_decimal', 0.0001)),
                      sl_pips=float(req.get('sl_pips', 50.0)),
                      lot=float(req.get('lot', 1.0)),
                      exchange_rate=float(req.get('exchange_rate', 1.0)))
        if obs.shape[0] != (args.window_len * n_feats + 6):
            sock.send_json({"ok": False, "error": "obs_dim_mismatch"})
            continue

        q = model(torch.from_numpy(obs).float().unsqueeze(0).to(device))
        a = int(q.argmax(dim=1).item())
        
        action_hist[a] += 1
        print("[serve] action_hist so far:", action_hist)
            
        # DEBUG: print every Nth step only if you want
        # (for now it’s okay to always print during a short test)
        print("[serve] a =", a, "q =", q.detach().cpu().numpy().tolist())
        
        out = {
            "ok": True,
            "a": a,
            "action": ACTIONS.get(a, "HOLD"),
            "q": q.squeeze(0).tolist(),
        }
        sock.send_json(out)
        
    sock.close(0)
    ctx.term()
    print("[serve] Closed socket and context. Bye.")

if __name__ == "__main__":
    main()
