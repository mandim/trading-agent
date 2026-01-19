import argparse
from typing import Optional

import numpy as np
import pandas as pd


def read_csv_autosep(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep=None, engine="python")
        if df.shape[1] == 1 and df.columns.size == 1 and ";" in str(df.columns[0]):
            raise ValueError("autosep_failed_semicolon")
        return df
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def parse_bar_time(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    s = s.str.replace("T", " ", regex=False)
    return pd.to_datetime(s, errors="coerce")


def read_ea_exec(path: str) -> pd.DataFrame:
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    if "bar_time" in df.columns:
        df["bar_time"] = parse_bar_time(df["bar_time"])
    if "close_time" in df.columns:
        df["close_time"] = parse_bar_time(df["close_time"])

    numeric_cols = [
        "bar_index",
        "action_req",
        "action_eff",
        "pos_side_before",
        "pos_side_after",
        "pos_age_bars",
        "min_hold_bars",
        "cooldown_before",
        "blocked_by_cooldown",
        "blocked_by_min_hold",
        "blocked_by_reverse",
        "order_send_attempted",
        "order_send_ok",
        "order_send_ticket",
        "order_send_error",
        "order_close_attempted",
        "order_close_ok",
        "order_close_error",
        "last_error",
        "close_detected",
        "close_ticket",
        "close_price",
        "close_profit",
        "close_swap",
        "close_commission",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bar_index" in df.columns:
        df = df.dropna(subset=["bar_index"]).copy()
        df["bar_index"] = df["bar_index"].astype(int)

    return df


def read_test_steps(path: str) -> pd.DataFrame:
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    if "bar_time" in df.columns:
        df["bar_time"] = parse_bar_time(df["bar_time"])

    numeric_cols = [
        "bar_index",
        "action_id",
        "action_requested",
        "action_effective",
        "position_side_before",
        "position_side",
        "pos_age_bars",
        "min_hold_bars",
        "cooldown_before",
        "blocked_by_cooldown",
        "blocked_by_min_hold",
        "blocked_by_reverse",
        "cooldown_remaining",
        "closed_trade",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bar_index" in df.columns:
        df = df.dropna(subset=["bar_index"]).copy()
        df["bar_index"] = df["bar_index"].astype(int)

    # Standardize action columns for comparison
    action_req = find_column(df, ["action_requested", "action_id", "action"])
    action_eff = find_column(df, ["action_effective"])

    if action_req is not None and action_req != "action_requested":
        df = df.rename(columns={action_req: "action_requested"})
    if action_eff is not None and action_eff != "action_effective":
        df = df.rename(columns={action_eff: "action_effective"})
    if "action_requested" in df.columns and "action_effective" not in df.columns:
        df["action_effective"] = df["action_requested"]

    return df


def normalize_reason(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip().lower()
    if "tp" in s:
        return "tp"
    if "sl" in s:
        return "sl"
    if "manual" in s or "reverse" in s or "eoe" in s:
        return "manual"
    return "unknown"


def align_on_bar_index(ea: pd.DataFrame, td: pd.DataFrame):
    candidates = []
    for shift in (-2, -1, 0, 1, 2):
        tmp = td.copy()
        tmp["bar_index"] = tmp["bar_index"] + shift
        merged = ea.merge(tmp, on="bar_index", suffixes=("_ea", "_td"))
        mismatch = None
        if "action_req_ea" in merged.columns and "action_requested_td" in merged.columns:
            mask = merged["action_req_ea"].notna() & merged["action_requested_td"].notna()
            mismatch = int((merged.loc[mask, "action_req_ea"] != merged.loc[mask, "action_requested_td"]).sum())
        candidates.append(
            {
                "shift": shift,
                "rows": len(merged),
                "mismatch": mismatch,
                "merged": merged,
            }
        )

    max_rows = max(c["rows"] for c in candidates)
    best = [c for c in candidates if c["rows"] == max_rows]

    # Prefer lower action mismatch if available
    best_with_mismatch = [c for c in best if c["mismatch"] is not None]
    if best_with_mismatch:
        chosen = min(best_with_mismatch, key=lambda c: c["mismatch"])
    else:
        chosen = best[0]

    if chosen["shift"] != 0:
        print(f"[align] Applying bar_index shift to TestDQN: {chosen['shift']}")
    if chosen["mismatch"] is not None:
        print(f"[align] action_req mismatches at chosen shift: {chosen['mismatch']}")

    return chosen["merged"]


def report_diff(m: pd.DataFrame, label: str, col_a: str, col_b: str) -> Optional[int]:
    if col_a not in m.columns or col_b not in m.columns:
        return None
    mask = m[col_a].notna() & m[col_b].notna()
    mismatches = int((m.loc[mask, col_a] != m.loc[mask, col_b]).sum())
    print(f"{label} mismatches: {mismatches}")
    return mismatches


def report_first_diffs(m: pd.DataFrame, cols: list[tuple[str, str]], max_rows: int):
    if "bar_index" not in m.columns:
        return
    diff_flags = []
    for col_a, col_b in cols:
        if col_a in m.columns and col_b in m.columns:
            diff_flags.append((m[col_a] != m[col_b]) & m[col_a].notna() & m[col_b].notna())
    if not diff_flags:
        return
    any_diff = diff_flags[0]
    for f in diff_flags[1:]:
        any_diff = any_diff | f
    subset = m[any_diff].copy()
    if subset.empty:
        return
    subset = subset.sort_values("bar_index")
    cols_out = ["bar_index"]
    if "bar_time_ea" in subset.columns:
        cols_out.append("bar_time_ea")
    for col_a, col_b in cols:
        if col_a in subset.columns and col_b in subset.columns:
            cols_out.extend([col_a, col_b])
    print("\nFirst divergences:")
    print(subset[cols_out].head(max_rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_exec", default="ea_exec_log.csv", help="EA execution log CSV")
    ap.add_argument("--td_steps", default="test_steps.csv", help="TestDQN steps CSV")
    ap.add_argument("--max_rows", type=int, default=10)
    args = ap.parse_args()

    ea = read_ea_exec(args.ea_exec)
    td = read_test_steps(args.td_steps)

    if "bar_index" in ea.columns and "bar_index" in td.columns:
        m = align_on_bar_index(ea, td)
    elif "bar_time" in ea.columns and "bar_time" in td.columns:
        m = ea.merge(td, on="bar_time", suffixes=("_ea", "_td"))
    else:
        raise RuntimeError("No common key to merge exec logs (bar_index or bar_time).")

    if len(m) == 0:
        raise RuntimeError("No overlapping rows after alignment.")

    print("Aligned rows:", len(m))

    report_diff(m, "action_requested", "action_req_ea", "action_requested_td")
    report_diff(m, "action_effective", "action_eff_ea", "action_effective_td")
    report_diff(m, "position_side_before", "pos_side_before_ea", "position_side_before_td")
    report_diff(m, "position_side_after", "pos_side_after_ea", "position_side_td")
    report_diff(m, "blocked_by_cooldown", "blocked_by_cooldown_ea", "blocked_by_cooldown_td")
    report_diff(m, "blocked_by_min_hold", "blocked_by_min_hold_ea", "blocked_by_min_hold_td")
    report_diff(m, "blocked_by_reverse", "blocked_by_reverse_ea", "blocked_by_reverse_td")
    report_diff(m, "cooldown_before", "cooldown_before_ea", "cooldown_before_td")
    report_diff(m, "pos_age_bars", "pos_age_bars_ea", "pos_age_bars_td")

    if "close_detected_ea" in m.columns and "closed_trade" in m.columns:
        m["close_detected_ea"] = pd.to_numeric(m["close_detected_ea"], errors="coerce")
        m["closed_trade_td"] = pd.to_numeric(m["closed_trade"], errors="coerce")
        report_diff(m, "closed_trade", "close_detected_ea", "closed_trade_td")

    if "close_reason_ea" in m.columns and "exit_reason" in m.columns:
        m["close_reason_ea_norm"] = m["close_reason_ea"].map(normalize_reason)
        m["exit_reason_td_norm"] = m["exit_reason"].map(normalize_reason)
        mask = (m.get("close_detected_ea") == 1) & (m.get("closed_trade") == 1)
        if mask.any():
            sub = m.loc[mask]
            mism = int((sub["close_reason_ea_norm"] != sub["exit_reason_td_norm"]).sum())
            print(f"close_reason mismatches (normalized): {mism}")

    if "order_send_attempted_ea" in m.columns:
        send_attempted = int(m["order_send_attempted_ea"].fillna(0).astype(int).sum())
        send_failed = int(
            ((m["order_send_attempted_ea"] == 1) & (m["order_send_ok_ea"] == 0)).sum()
        )
        print(f"order_send_attempted: {send_attempted} failed: {send_failed}")
        if send_failed > 0 and "order_send_error_ea" in m.columns:
            errs = m.loc[m["order_send_attempted_ea"] == 1, "order_send_error_ea"].dropna()
            if not errs.empty:
                top = errs.value_counts().head(5).to_string()
                print("order_send_error top codes:")
                print(top)

    if "order_close_attempted_ea" in m.columns:
        close_attempted = int(m["order_close_attempted_ea"].fillna(0).astype(int).sum())
        close_failed = int(
            ((m["order_close_attempted_ea"] == 1) & (m["order_close_ok_ea"] == 0)).sum()
        )
        print(f"order_close_attempted: {close_attempted} failed: {close_failed}")
        if close_failed > 0 and "order_close_error_ea" in m.columns:
            errs = m.loc[m["order_close_attempted_ea"] == 1, "order_close_error_ea"].dropna()
            if not errs.empty:
                top = errs.value_counts().head(5).to_string()
                print("order_close_error top codes:")
                print(top)

    diff_cols = [
        ("action_req_ea", "action_requested_td"),
        ("action_eff_ea", "action_effective_td"),
        ("pos_side_before_ea", "position_side_before_td"),
        ("pos_side_after_ea", "position_side_td"),
        ("blocked_by_cooldown_ea", "blocked_by_cooldown_td"),
        ("blocked_by_min_hold_ea", "blocked_by_min_hold_td"),
        ("blocked_by_reverse_ea", "blocked_by_reverse_td"),
        ("cooldown_before_ea", "cooldown_before_td"),
        ("close_detected_ea", "closed_trade"),
    ]
    report_first_diffs(m, diff_cols, args.max_rows)


if __name__ == "__main__":
    main()
