import os, json, time, argparse, numpy as np, torch, torch.nn as nn
import zmq

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
              price_norms):
    """
    bar_window: shape (L, n_feats), oldest->newest
    price_norms: dict of RollingNorm for 'ask','bid','spread'
    """
    # normalize bar features if scalers exist (train split stats)
    if bars_mean is not None and bars_std is not None:
        bar_window = (bar_window - bars_mean) / (bars_std + 1e-8)

    spread = ask - bid
    # update rolling stats then normalize
    price_norms["ask"].update(ask)
    price_norms["bid"].update(bid)
    price_norms["spread"].update(spread)

    a = price_norms["ask"].normalize(ask)
    b = price_norms["bid"].normalize(bid)
    s = price_norms["spread"].normalize(spread)

    obs = np.hstack([bar_window.ravel(), np.array([a, b, s], dtype=np.float32)]).astype(np.float32)
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)  # match env safeguards
    return obs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default="cache_fx_EURUSD_D1_2020_2025")
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

        # Required fields
        bar_window = req.get("bar_window")  # list[list[float]] shape (L, n_feats)
        ask = req.get("ask")
        bid = req.get("bid")

        if bar_window is None or ask is None or bid is None:
            sock.send_json({"ok": False, "error": "missing_fields"})
            continue

        bar_window = np.asarray(bar_window, dtype=np.float32)
        if bar_window.ndim != 2 or bar_window.shape[0] != args.window_len:
            sock.send_json({"ok": False, "error": "bad_bar_window_shape"})
            continue

        if n_feats is None:
            n_feats = int(bar_window.shape[1])
            obs_dim = args.window_len * n_feats + 3

            # init model now that we know obs_dim
            model = QNet(obs_dim, 3).to(device)
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            print(f"[serve] Model ready: obs_dim={obs_dim}, n_feats={n_feats}")

        obs = build_obs(bar_window, bars_mean, bars_std, float(ask), float(bid), price_norms)
        if obs.shape[0] != (args.window_len * n_feats + 3):
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
