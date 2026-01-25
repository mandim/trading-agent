import os, json, csv, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from trading_env_merged import TradingEnv


# -------------------- QNet (MUST match training) --------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# -------------------- Env factory (MATCH training config) --------------------
def make_test_env(
    seed: int,
    cache_dir: str,
    eval_start_date: str,
    eval_end_date: str,
    commission_per_lot_per_side_usd: float,
    enable_commission: bool,
    enable_swaps: bool,
    swap_long_pips_per_day: float,
    swap_short_pips_per_day: float,
    slippage_pips_open: float,
    slippage_pips_close: float,
    slippage_mode: str,
    enable_slippage: bool,
    slippage_pips: float,
    other_fixed_cost_per_trade_usd: float,
    reset_balance_each_episode: bool = False,
    max_steps_per_episode: int | None = None,   # None => run full window unless env hits end
):
    return TradingEnv(
        pip_decimal=0.0001,
        candles_file="unused.csv",
        tick_file="unused.csv",
        cache_dir=cache_dir,

        # Trading (MATCH training)
        tp_pips=50.0,
        sl_pips=40.0,
        lot=1.0,
        start_balance=100_000.0,
        reset_balance_each_episode=reset_balance_each_episode,
        exchange_rate=1.0,

        # Costs (MATCH training)
        commission_per_lot_per_side_usd=commission_per_lot_per_side_usd,
        enable_commission=enable_commission,
        enable_swaps=enable_swaps,
        swap_long_pips_per_day=swap_long_pips_per_day,
        swap_short_pips_per_day=swap_short_pips_per_day,
        slippage_pips_open=slippage_pips_open,
        slippage_pips_close=slippage_pips_close,
        slippage_mode=slippage_mode,
        enable_slippage=enable_slippage,
        slippage_pips=slippage_pips,
        other_fixed_cost_per_trade_usd=other_fixed_cost_per_trade_usd,

        # Risk/penalties (MATCH training)
        dd_penalty_lambda=0.0,
        max_dd_stop=0.30,
        flat_penalty_R=0.0,
        trade_penalty_R=0.2,          # ONLY keep this if training env used same value!
        min_hold_bars=5,
        allow_reverse=False,
        cooldown_bars_after_close=3,

        # Episodes / split
        max_steps_per_episode=max_steps_per_episode,
        eval_mode=True,

        # Date-based window (STRICT TEST WINDOW)
        train_start_date="2019-01-01",     # only affects scaler fit if you do it that way
        eval_start_date=eval_start_date,
        eval_end_date=eval_end_date,       # exclusive end in your env logic

        # Obs / normalization
        window_len=32,
        normalize_prices=True,
        normalize_bars=True,
        use_prev_bar_features=True,

        # Runtime
        seed=seed,
    )


def profit_factor(gp: float, gl: float) -> float:
    return float("inf") if gl <= 0 else float(gp / gl)

def format_bar_time(bt) -> str:
    if bt is None:
        return ""

    # numpy datetime64
    if isinstance(bt, np.datetime64):
        s = str(bt.astype("datetime64[m]"))  # YYYY-MM-DDTHH:MM
        return s.replace("-", ".").replace("T", " ")

    # epoch seconds (int/float or numpy scalar)
    if isinstance(bt, (int, float, np.integer, np.floating)):
        # bt appears to be UNIX seconds
        dt = datetime.utcfromtimestamp(float(bt))
        return dt.strftime("%Y.%m.%d %H:%M")

    # python datetime
    if hasattr(bt, "strftime"):
        return bt.strftime("%Y.%m.%d %H:%M")

    return str(bt)

# -------------------- Strict single-pass test --------------------
def run_strict_test(policy, env, device, tb: SummaryWriter | None):

    action_rows = []
    parity_rows = []

    s, _ = env.reset(seed=getattr(env, "_seed", None) or 999)
    done = False

    pip_value = float(env._pip_value_usd())
    one_R_usd = max(1e-9, pip_value * float(env.sl_pips))

    # Aggregates
    steps = 0
    trades = 0
    wins = 0
    sum_pnl_pips = 0.0

    gross_profit = 0.0
    gross_loss = 0.0
    net_profit_pos = 0.0
    net_loss_pos = 0.0

    cost_comm = 0.0
    cost_swap = 0.0
    cost_other = 0.0

    max_dd = 0.0
    ep_ret = 0.0

    # Logs
    trade_rows = []
    step_rows = []

    with torch.no_grad():
        while not done:
            q = policy(torch.from_numpy(s).float().unsqueeze(0).to(device))
            a = int(q.argmax(dim=1).item())

            # Parity log inputs at decision time (pre-step)
            decision_tick = int(env.t)
            decision_bar_index = int(env.tick_to_bar[decision_tick])
            bar_time_decision = format_bar_time(env.bar_times[decision_bar_index])

            ask = float(env.tick_ask[decision_tick])
            bid = float(env.tick_bid[decision_tick])

            position_side = int(env.position_side)
            pos_side_before = int(env.position_side)
            cooldown_before = int(getattr(env, "cooldown_remaining", 0))
            min_hold_bars = int(getattr(env, "min_hold_bars", 0))
            entry_ask = env.position_entry_ask if env.position_entry_ask is not None else None
            entry_bid = env.position_entry_bid if env.position_entry_bid is not None else None
            pos_age_bars = 0.0
            if position_side != 0 and env.position_entry_idx is not None:
                entry_bar = int(env.tick_to_bar[int(env.position_entry_idx)])
                pos_age_bars = max(0, decision_bar_index - entry_bar)

            feat_vals = {}
            prev_bar_idx = max(0, decision_bar_index - 1)
            if hasattr(env, "bar_features"):
                feats = np.asarray(env.bar_features[prev_bar_idx], dtype=np.float32)
                if feats.size == 15:
                    feat_vals = {f"feat_{i}": float(feats[i]) for i in range(15)}

            parity_rows.append({
                "bar_index": decision_bar_index,
                "bar_time": bar_time_decision,
                "action": a,
                "ask": ask,
                "bid": bid,
                "position_side": position_side,
                "entry_ask": entry_ask,
                "entry_bid": entry_bid,
                "pos_age_bars": float(pos_age_bars),
                "sl_pips": float(env.sl_pips),
                "lot": float(env.lot),
                "exchange_rate": float(env.exchange_rate),
                **feat_vals,
            })

            s, r, term, trunc, info = env.step(a)
            done = bool(term or trunc)
            
            bar_index = int(info.get("bar_index", -1))
            if bar_index < 0:
                # termination step or empty info -> don't log
                continue

            bar_time = format_bar_time(env.bar_times[bar_index])

            action_rows.append({
                "bar_index": bar_index,
                "bar_time": bar_time,
                "action": a,
                "position_side": int(info.get("position_side", 0)),
            })

            steps += 1
            ep_ret += float(r)
            max_dd = max(max_dd, float(info.get("max_drawdown", 0.0)))

            # per-step log (equity curve)
            step_rows.append({
                "step": steps,
                "tick_index": int(info.get("t", -1)),
                "bar_index": int(info.get("bar_index", -1)),
                "action_id": int(a),
                "action_requested": int(info.get("action_requested", a)),
                "action_effective": int(info.get("action_effective", a)),
                "position_side_before": int(pos_side_before),
                "position_side": int(info.get("position_side", 0)),
                "pos_age_bars": float(pos_age_bars),
                "min_hold_bars": int(min_hold_bars),
                "cooldown_before": int(cooldown_before),
                "blocked_by_cooldown": int(bool(info.get("blocked_by_cooldown", False))),
                "blocked_by_min_hold": int(bool(info.get("blocked_by_min_hold", False))),
                "blocked_by_reverse": int(bool(info.get("blocked_by_reverse", False))),
                "cooldown_remaining": int(info.get("cooldown_remaining", 0)),
                "equity": float(info.get("equity", float("nan"))),
                "balance": float(info.get("balance", float("nan"))),
                "reward": float(r),
                "max_drawdown": float(info.get("max_drawdown", 0.0)),
                "opened_trade": int(bool(info.get("opened_trade", False))),
                "closed_trade": int(bool(info.get("closed_trade", False))),
                "exit_reason": info.get("exit_reason", ""),
            })

            if tb is not None and steps % 10 == 0:
                if "equity" in info:
                    tb.add_scalar("test/equity", float(info["equity"]), steps)
                tb.add_scalar("test/reward", float(r), steps)
                tb.add_scalar("test/max_drawdown", float(info.get("max_drawdown", 0.0)), steps)

            # per-trade log
            if info.get("closed_trade", False):
                trades += 1

                gross_usd = float(info.get("profit", 0.0))
                pnl_pips = float(info.get("pnl_pips", 0.0))
                comm = float(info.get("commission_usd", 0.0))
                swap = float(info.get("swap_usd", 0.0))
                other = float(info.get("other_cost_usd", 0.0))

                net_usd = gross_usd - comm - swap - other
                net_R = net_usd / one_R_usd

                sum_pnl_pips += pnl_pips
                if net_usd >= 0:
                    wins += 1

                # gross
                if gross_usd >= 0:
                    gross_profit += gross_usd
                else:
                    gross_loss += -gross_usd

                # net
                if net_usd >= 0:
                    net_profit_pos += net_usd
                else:
                    net_loss_pos += -net_usd

                cost_comm += comm
                cost_swap += swap
                cost_other += other

                close_time = ""
                if bar_index >= 0 and bar_index < len(env.bar_times):
                    close_time = format_bar_time(env.bar_times[bar_index])

                trade_rows.append({
                    "trade_idx": trades,
                    "bar_index": int(info.get("bar_index", -1)),
                    "tick_index": int(info.get("t", -1)),
                    "close_time": close_time,
                    "exit_reason": info.get("exit_reason", ""),
                    "pnl_pips": pnl_pips,
                    "profit_gross_usd": gross_usd,
                    "profit_net_usd": net_usd,
                    "profit_net_R": net_R,
                    "commission_usd": comm,
                    "swap_usd": swap,
                    "other_cost_usd": other,
                    "equity_after": float(info.get("equity", float("nan"))),
                    "balance_after": float(info.get("balance", float("nan"))),
                    "max_drawdown": float(info.get("max_drawdown", 0.0)),
                })

                if tb is not None:
                    tb.add_scalar("test/trade_net_usd", net_usd, trades)
                    tb.add_scalar("test/trade_net_R", net_R, trades)
                    tb.add_scalar("test/trade_pnl_pips", pnl_pips, trades)

    summary = {
        "steps": int(steps),
        "ep_return": float(ep_ret),

        "trades": int(trades),
        "win_rate": 0.0 if trades == 0 else float(wins / trades),
        "avg_pnl_pips": 0.0 if trades == 0 else float(sum_pnl_pips / trades),

        "profit_factor_gross": profit_factor(gross_profit, gross_loss),
        "profit_factor_net": profit_factor(net_profit_pos, net_loss_pos),

        "net_profit_usd": float(net_profit_pos - net_loss_pos),
        "gross_profit_usd": float(gross_profit),
        "gross_loss_usd": float(gross_loss),

        "commission_total_usd": float(cost_comm),
        "swap_total_usd": float(cost_swap),
        "other_costs_total_usd": float(cost_other),

        "max_drawdown": float(max_dd),
    }
    return summary, trade_rows, step_rows, action_rows, parity_rows


def _write_csv(path: str, rows: list[dict]):
    dirn = os.path.dirname(path)
    if dirn:  # only if a directory part exists
        os.makedirs(dirn, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _coerce_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "y", "on"):
            return True
        if v in ("0", "false", "no", "n", "off", ""):
            return False
    raise ValueError(f"{key} must be a boolean")


def load_config(config_path: str) -> argparse.Namespace:
    defaults = {
        "model": "models/dqn_best.pt",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cache_dir": "cache_fx_EURUSD_D1_fx",
        "seed": 999,
        "start": "2024-01-01",
        "end": "2025-01-01",
        "steps_csv": "test_steps.csv",
        "trades_csv": "test_trades_2025.csv",
        "json": "test_summary.json",
        "tb_dir": "runs/test",
        "no_tb": True,
        "reset_balance_each_episode": False,
        "max_steps": 0,
        "actions_csv": "test_actions_2025.csv",
        "parity_log": "td_parity_log.csv",
        "commission_per_lot_per_side_usd": 0.0,
        "enable_commission": False,
        "enable_swaps": True,
        "swap_long_pips_per_day": -0.971,
        "swap_short_pips_per_day": 0.45,
        "slippage_pips_open": 0.0,
        "slippage_pips_close": 0.0,
        "slippage_mode": "fixed",
        "enable_slippage": False,
        "slippage_pips": 0.0,
        "other_fixed_cost_per_trade_usd": 0.0,
    }

    if not os.path.exists(config_path):
        raise SystemExit(f"[test] Missing config file: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[test] Invalid JSON in {config_path}: {e}")

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"[test] Config must be a JSON object, got {type(cfg).__name__}")

    merged = defaults.copy()
    merged.update(cfg)

    try:
        merged["seed"] = int(merged["seed"])
        merged["max_steps"] = int(merged["max_steps"])
        merged["commission_per_lot_per_side_usd"] = float(merged["commission_per_lot_per_side_usd"])
        merged["swap_long_pips_per_day"] = float(merged["swap_long_pips_per_day"])
        merged["swap_short_pips_per_day"] = float(merged["swap_short_pips_per_day"])
        merged["slippage_pips_open"] = float(merged["slippage_pips_open"])
        merged["slippage_pips_close"] = float(merged["slippage_pips_close"])
        merged["slippage_pips"] = float(merged["slippage_pips"])
        merged["other_fixed_cost_per_trade_usd"] = float(merged["other_fixed_cost_per_trade_usd"])
        merged["no_tb"] = _coerce_bool(merged["no_tb"], "no_tb")
        merged["reset_balance_each_episode"] = _coerce_bool(
            merged["reset_balance_each_episode"], "reset_balance_each_episode"
        )
        merged["enable_commission"] = _coerce_bool(merged["enable_commission"], "enable_commission")
        merged["enable_swaps"] = _coerce_bool(merged["enable_swaps"], "enable_swaps")
        merged["enable_slippage"] = _coerce_bool(merged["enable_slippage"], "enable_slippage")
    except (TypeError, ValueError) as e:
        raise SystemExit(f"[test] Bad config value: {e}")

    for key in (
        "model", "device", "cache_dir", "start", "end", "steps_csv", "trades_csv", "json",
        "tb_dir", "actions_csv", "parity_log", "slippage_mode",
    ):
        if merged.get(key) is not None:
            merged[key] = str(merged[key])

    device_val = merged.get("device")
    if device_val is None or str(device_val).strip().lower() == "auto":
        merged["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return argparse.Namespace(**merged)


def main():
    config_path = os.getenv("TEST_DQN_CONFIG", "test_dqn_config.json")
    args = load_config(config_path)
    print(f"[test] Using config: {config_path}")

    device = torch.device(args.device)

    max_steps = None if args.max_steps == 0 else int(args.max_steps)

    env = make_test_env(
        seed=args.seed,
        cache_dir=args.cache_dir,
        eval_start_date=args.start,
        eval_end_date=args.end,
        commission_per_lot_per_side_usd=args.commission_per_lot_per_side_usd,
        enable_commission=bool(args.enable_commission),
        enable_swaps=bool(args.enable_swaps),
        swap_long_pips_per_day=args.swap_long_pips_per_day,
        swap_short_pips_per_day=args.swap_short_pips_per_day,
        slippage_pips_open=args.slippage_pips_open,
        slippage_pips_close=args.slippage_pips_close,
        slippage_mode=args.slippage_mode,
        enable_slippage=bool(args.enable_slippage),
        slippage_pips=args.slippage_pips,
        other_fixed_cost_per_trade_usd=args.other_fixed_cost_per_trade_usd,
        reset_balance_each_episode=args.reset_balance_each_episode,
        max_steps_per_episode=max_steps,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy = QNet(obs_dim, n_actions).to(device)
    policy.load_state_dict(torch.load(args.model, map_location=device))
    policy.eval()

    tb = None
    tb_dir = None
    if not args.no_tb:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir = os.path.join(args.tb_dir, f"{stamp}_STRICT_TEST")
        os.makedirs(tb_dir, exist_ok=True)
        tb = SummaryWriter(tb_dir)

        tb.add_hparams(
            {
                "model": os.path.basename(args.model),
                "seed": args.seed,
                "start": args.start,
                "end": args.end,
                "reset_balance_each_episode": int(args.reset_balance_each_episode),
                "max_steps": int(args.max_steps),
                "commission_per_lot_per_side_usd": float(args.commission_per_lot_per_side_usd),
                "enable_commission": int(args.enable_commission),
                "enable_swaps": int(args.enable_swaps),
                "swap_long_pips_per_day": float(args.swap_long_pips_per_day),
                "swap_short_pips_per_day": float(args.swap_short_pips_per_day),
                "slippage_pips_open": float(args.slippage_pips_open),
                "slippage_pips_close": float(args.slippage_pips_close),
                "slippage_mode": str(args.slippage_mode),
                "enable_slippage": int(args.enable_slippage),
                "slippage_pips": float(args.slippage_pips),
                "other_fixed_cost_per_trade_usd": float(args.other_fixed_cost_per_trade_usd),
            },
            {},
        )

    summary, trade_rows, step_rows, action_rows, parity_rows = run_strict_test(policy, env, device, tb)

    _write_csv(args.trades_csv, trade_rows)
    _write_csv(args.steps_csv, step_rows)
    
    _write_csv(args.actions_csv, action_rows)
    print(f"Saved actions CSV -> {args.actions_csv}")

    if args.parity_log:
        _write_csv(args.parity_log, parity_rows)
        print(f"Saved parity CSV -> {args.parity_log}")

    json_dir = os.path.dirname(args.json)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 72)
    print("[STRICT TEST SUMMARY]")
    print(json.dumps(summary, indent=2))
    print(f"Saved steps CSV  -> {args.steps_csv}")
    print(f"Saved trades CSV -> {args.trades_csv}")
    print(f"Saved JSON       -> {args.json}")
    if tb is not None:
        print(f"TensorBoard dir  -> {tb_dir}")
        tb.close()


if __name__ == "__main__":
    main()
