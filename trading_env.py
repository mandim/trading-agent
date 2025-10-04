import pandas as pd
from server import ZMQRepServer
import numpy as np
import os

class TradingEnv:

    def __init__(self, pip_decimal: float, candles_file: str, tick_file: str, bind_address="tcp://*:5555", cache_dir: str = "cache_fx_EURUSD_D1"):
        self.candles_file = candles_file
        self.tick_file = tick_file
        self.current_tick_row = 0
        self.position_open = False
        self.server = ZMQRepServer(bind_address, self._handle_request)
        self.pip_decimal = pip_decimal
        self.tp_pips = 50.0
        self.sl_pips = 50.0
        self.lot = 1.0
        self.balance = 100000.0
        self.truncated = False
        self.done = False
        self.risk_per_trade_usd = 1000.0   # denominator for PnL normalization (e.g., ~1% of 100k)
        self.dd_penalty_lambda = 1.0       # how strongly to punish new drawdown (tune 0.3–2.0)
        self.max_dd_stop = 0.30            # hard stop at 30% drawdown (optional)
        
        # RUNTIME (also set these in reset)
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0

        # ---- BAR/TICK FUSION CONFIG ----
        self.window_len = 32  # number of bars to include in the observation window

        self.cache_dir = cache_dir
        self._load_cache(self.cache_dir)

        # Kept-tick cursor (0..n_ticks-1). We’ll set a proper start in reset()
        self.t = 0

    def start_server(self):
        print("Starting Env Server...")
        self.server.start()
    
    def stop_server(self):
        print("Stoping Env Server...")
        self.server.stop()

    def _handle_request(self, request):
        """
        Process incoming request and return a reply.
        Request is expected to be a dict; adapt as needed.
        """
        # Simple example: support {"cmd": "ping"} and {"cmd": "sum", "values": [..]}
        cmd = request.get("cmd") if isinstance(request, dict) else None

        if cmd == "BUY":
            self.step("BUY")
            return {"reply": "BUY position closed and return current state to the Agent"}
        elif cmd == "SELL":
            self.step("SELL")
            return {"reply": "SELL position closed and return current state to the Agent"}
        elif cmd == "HOLD":
            return {"reply": "Hold position and continue"}
        else:
            # default: echo
            return {"reply": "unknown_command", "received": request}
    
    def _load_cache(self, cache_dir: str):
        needed = ["bars_features.npy", "bar_times.npy", "tick_ask.npy", "tick_bid.npy", "tick_to_bar.npy"]
        for f in needed:
            p = os.path.join(cache_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing cache artifact: {p}")

        self.bar_features = np.load(os.path.join(cache_dir, "bars_features.npy"), mmap_mode="r")   # (n_bars, n_feats) float32
        self.bar_times    = np.load(os.path.join(cache_dir, "bar_times.npy"),    mmap_mode="r")    # (n_bars,) int64
        self.tick_ask     = np.load(os.path.join(cache_dir, "tick_ask.npy"),     mmap_mode="r")    # (n_ticks,) float32
        self.tick_bid     = np.load(os.path.join(cache_dir, "tick_bid.npy"),     mmap_mode="r")    # (n_ticks,) float32
        self.tick_to_bar  = np.load(os.path.join(cache_dir, "tick_to_bar.npy"),  mmap_mode="r")    # (n_ticks,) int32

        self.n_bars  = int(self.bar_features.shape[0])
        self.n_feats = int(self.bar_features.shape[1])
        self.n_ticks = int(self.tick_ask.shape[0])

        assert self.tick_bid.shape == self.tick_ask.shape
        assert self.tick_to_bar.shape[0] == self.n_ticks
    
    def reset(self):
        self.balance = 100000.0
        self.position_open = False
        self.truncated = False
        self.done = False
        self.equity = self.balance
        self.equity_peak = self.balance
        self.max_drawdown = 0.0

        # warm start: first tick whose bar index has >= window_len bars behind it
        min_bar = self.window_len - 1
        # tick_to_bar is non-decreasing; find first tick mapping to bar >= min_bar
        self.t = int(np.searchsorted(self.tick_to_bar, min_bar, side="left"))
        if self.t >= self.n_ticks:
            raise ValueError("No valid ticks after warmup. Check data/time parsing and warmup settings.")

        # maintain current_tick_row for backward compatibility (just mirror kept tick index)
        self.current_tick_row = self.t

        obs = self._get_observation(self.t)
        return obs, {}
    
    def step(self, action: str):
        if self.done:
            obs = self._get_observation(self.t)
            return obs, 0.0, True, self.truncated, {"note": "episode already done"}

        # guard: end of data
        if self.t >= self.n_ticks - 1:
            self.done = True
            obs = self._get_observation(self.t)
            return obs, 0.0, True, self.truncated, {"reason": "eof"}

        tp_value = sl_value = None
        is_profitable = None

        if action == "HOLD":
            self.t += 1
            self.current_tick_row = self.t
            obs = self._get_observation(self.t)
            return obs, 0.0, False, self.truncated, {"action": "HOLD"}

        # entry prices at current tick
        entry_ask = float(self.tick_ask[self.t])
        entry_bid = float(self.tick_bid[self.t])

        if action == "BUY":
            entry = entry_ask
            tp_value = entry + self.tp_pips * self.pip_decimal
            sl_value = entry - self.sl_pips * self.pip_decimal

            k = self.t
            # long exits at BID
            while k + 1 < self.n_ticks:
                k += 1
                px = float(self.tick_bid[k])
                if px >= tp_value:
                    is_profitable = True
                    break
                if px <= sl_value:
                    is_profitable = False
                    break
            self.t = k

            # optional: subtract entry spread once
            pnl_pips = (float(self.tick_bid[self.t]) - entry) / self.pip_decimal
            pnl_pips -= (entry_ask - entry_bid) / self.pip_decimal

        elif action == "SELL":
            entry = entry_bid
            tp_value = entry - self.tp_pips * self.pip_decimal
            sl_value = entry + self.sl_pips * self.pip_decimal

            k = self.t
            # short exits at ASK
            while k + 1 < self.n_ticks:
                k += 1
                px = float(self.tick_ask[k])
                if px <= tp_value:
                    is_profitable = True
                    break
                if px >= sl_value:
                    is_profitable = False
                    break
            self.t = k

            # optional: subtract entry spread once
            pnl_pips = (entry - float(self.tick_ask[self.t])) / self.pip_decimal
            pnl_pips -= (entry_ask - entry_bid) / self.pip_decimal

        else:
            # Unknown action: treat as HOLD
            self.t += 1
            self.current_tick_row = self.t
            obs = self._get_observation(self.t)
            return obs, 0.0, False, self.truncated, {"action": action, "note": "unknown_action_noop"}

        # EOF without TP/SL? treat as SL (your previous policy)
        if is_profitable is None:
            is_profitable = False
            if self.t >= self.n_ticks - 1:
                self.done = True

        # reward & equity updates (reuse your existing logic)
        reward = self._calculate_reward(isProfitable=is_profitable)
        # If you prefer reward from pnl, use pnl_pips above.
        # equity updates if you keep them:
        # self.equity += self._usd_from_pips(pnl_pips)

        self.equity_peak = max(self.equity_peak, self.equity)
        dd = 1.0 - (self.equity / (self.equity_peak + 1e-9))
        self.max_drawdown = max(self.max_drawdown, dd)
        if dd >= self.max_dd_stop:
            self.done = True
            self.truncated = True

        self.current_tick_row = self.t
        obs = self._get_observation(self.t)
        info = {
            "action": action,
            "tp": tp_value,
            "sl": sl_value,
            "final_idx": self.t,
            "balance": self.balance,
            "equity": self.equity,
            "equity_peak": self.equity_peak,
            "max_drawdown": self.max_drawdown,
            "is_profitable": is_profitable,
            "last_reward": reward,
        }
        return obs, float(reward), self.done, self.truncated, info

    def _get_observation(self, tick_idx: int):
        """
        Build observation at kept tick index 'tick_idx':
        - live tick features: ask, bid, spread
        - window_len bars of precomputed indicators up to the most recent CLOSED bar
        """
        bar_idx = int(self.tick_to_bar[tick_idx])
        L = self.window_len
        start = bar_idx - L + 1

        if start < 0:
            pad = np.zeros((-start, self.n_feats), dtype=np.float32)
            win = np.asarray(self.bar_features[0:bar_idx+1], dtype=np.float32)
            feat_win = np.vstack([pad, win])
        else:
            feat_win = np.asarray(self.bar_features[start:bar_idx+1], dtype=np.float32)

        ask = float(self.tick_ask[tick_idx])
        bid = float(self.tick_bid[tick_idx])
        spread = ask - bid

        obs = np.hstack([
            feat_win.ravel(),
            np.array([ask, bid, spread, float(self.position_open)], dtype=np.float32),
        ])
        # safety
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    def _calculate_reward(self, isProfitable: bool):
        # --- 1) Compute trade PnL ---
        old_balance = self.balance
        exchange_rate = 1.0  # adjust if needed
        pip_value = ((self.lot * 100000) * self.pip_decimal) / exchange_rate
        profit = 0.0
        if isProfitable is True:
            profit = pip_value * self.tp_pips
        elif isProfitable is False:
            profit = -pip_value * self.sl_pips

        # --- 2) Update balance & equity ---
        next_balance = old_balance + profit
        if next_balance <= 0:
            self.truncated = True
            self.done = True
            self.balance = 0.0
        else:
            self.balance = next_balance

        old_equity = self.equity
        self.equity = self.balance  # equity = balance here (no floating PnL)

        # --- 3) Update drawdown metrics ---
        if self.equity > self.equity_peak:
            self.equity_peak = self.equity
        dd_now = 0.0 if self.equity_peak == 0 else (self.equity_peak - self.equity) / self.equity_peak
        dd_now = max(0.0, dd_now)
        new_dd_increment = max(0.0, dd_now - self.max_drawdown)
        if dd_now > self.max_drawdown:
            self.max_drawdown = dd_now

        # --- 4) Normalize profit and apply DD penalty ---
        denom = max(1e-8, self.risk_per_trade_usd)
        pnl_norm = profit / denom
        penalty = self.dd_penalty_lambda * new_dd_increment
        reward = pnl_norm - penalty

        # --- 5) Hard stop if max DD breached ---
        if self.max_drawdown >= self.max_dd_stop:
            self.done = True
            self.truncated = True

        return float(reward)
