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


def parse_bar_time(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    s = s.str.replace("T", " ", regex=False)
    return pd.to_datetime(s, errors="coerce")


def read_parity_log(path: str) -> pd.DataFrame:
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    if "bar_time" in df.columns:
        df["bar_time"] = parse_bar_time(df["bar_time"])

    for col in [
        "bar_index",
        "action",
        "ask",
        "bid",
        "position_side",
        "entry_ask",
        "entry_bid",
        "pos_age_bars",
        "sl_pips",
        "lot",
        "exchange_rate",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "bar_index" in df.columns:
        df = df.dropna(subset=["bar_index"]).copy()
        df["bar_index"] = df["bar_index"].astype(int)

    return df


def _mean_price_diff(m: pd.DataFrame) -> Optional[float]:
    diffs = []
    for field in ("ask", "bid"):
        col_a = f"{field}_ea"
        col_b = f"{field}_td"
        if col_a in m.columns and col_b in m.columns:
            d = (m[col_a] - m[col_b]).abs()
            d = d.dropna()
            if not d.empty:
                diffs.append(d)
    if not diffs:
        return None
    combo = pd.concat(diffs, axis=1)
    return float(combo.mean(axis=1).mean())


def align_on_bar_index(ea: pd.DataFrame, td: pd.DataFrame):
    candidates = []
    for shift in (-2, -1, 0, 1, 2):
        tmp = td.copy()
        tmp["bar_index"] = tmp["bar_index"] + shift
        merged = ea.merge(tmp, on="bar_index", suffixes=("_ea", "_td"))
        candidates.append(
            {
                "shift": shift,
                "rows": len(merged),
                "price_diff": _mean_price_diff(merged),
                "merged": merged,
            }
        )

    max_rows = max(c["rows"] for c in candidates)
    best = [c for c in candidates if c["rows"] == max_rows]

    # If price diff available, choose the smallest among max overlap.
    best_with_price = [c for c in best if c["price_diff"] is not None]
    if best_with_price:
        chosen = min(best_with_price, key=lambda c: c["price_diff"])
    else:
        chosen = best[0]

    if chosen["shift"] != 0:
        print(f"[align] Applying bar_index shift to TestDQN: {chosen['shift']}")
    if chosen["price_diff"] is not None:
        print(f"[align] mean price diff for chosen shift: {chosen['price_diff']:.10f}")

    return chosen["merged"]


def report_price_diff(m: pd.DataFrame, field: str, pip_decimal: float, max_rows: int):
    col_a = f"{field}_ea"
    col_b = f"{field}_td"
    if col_a not in m.columns or col_b not in m.columns:
        return
    diff = (m[col_a] - m[col_b]).abs()
    diff = diff.dropna()
    if diff.empty:
        return
    print(
        f"{field} diff: mean={diff.mean():.10f} max={diff.max():.10f} "
        f"(pips mean={diff.mean()/pip_decimal:.6f} max={diff.max()/pip_decimal:.6f})"
    )
    top = m.assign(_diff=diff).sort_values("_diff", ascending=False).head(max_rows)
    cols = ["bar_index"]
    if "bar_time_ea" in m.columns:
        cols.append("bar_time_ea")
    cols += [col_a, col_b, "_diff"]
    print(top[cols].to_string(index=False))


def report_numeric_diff(m: pd.DataFrame, field: str, max_rows: int):
    col_a = f"{field}_ea"
    col_b = f"{field}_td"
    if col_a not in m.columns or col_b not in m.columns:
        return
    diff = (m[col_a] - m[col_b]).abs()
    diff = diff.dropna()
    if diff.empty:
        return
    print(f"{field} diff: mean={diff.mean():.6f} max={diff.max():.6f}")
    top = m.assign(_diff=diff).sort_values("_diff", ascending=False).head(max_rows)
    cols = ["bar_index"]
    if "bar_time_ea" in m.columns:
        cols.append("bar_time_ea")
    cols += [col_a, col_b, "_diff"]
    print(top[cols].to_string(index=False))


def report_conditional_diffs(m: pd.DataFrame, pip_decimal: float, max_rows: int):
    if "position_side_ea" not in m.columns or "position_side_td" not in m.columns:
        return

    same_pos = (m["position_side_ea"] == m["position_side_td"]) & (m["position_side_ea"] != 0)
    if same_pos.any():
        sub = m[same_pos]
        print(f"rows with same non-flat position: {len(sub)}")
        report_price_diff(sub, "entry_ask", pip_decimal, max_rows)
        report_price_diff(sub, "entry_bid", pip_decimal, max_rows)
        report_numeric_diff(sub, "pos_age_bars", max_rows)
    else:
        print("rows with same non-flat position: 0")


def report_mismatch_cross(m: pd.DataFrame, max_rows: int):
    if "action_ea" in m.columns and "action_td" in m.columns and \
       "position_side_ea" in m.columns and "position_side_td" in m.columns:
        action_same = m["action_ea"] == m["action_td"]
        pos_same = m["position_side_ea"] == m["position_side_td"]

        mismatch_action_pos_same = m[~action_same & pos_same]
        mismatch_pos_action_same = m[~pos_same & action_same]

        print(f"action mismatches with same position: {len(mismatch_action_pos_same)}")
        if len(mismatch_action_pos_same) > 0:
            cols = ["bar_index"]
            if "bar_time_ea" in m.columns:
                cols.append("bar_time_ea")
            cols += ["action_ea", "action_td", "position_side_ea"]
            print(mismatch_action_pos_same[cols].head(max_rows).to_string(index=False))

        print(f"position mismatches with same action: {len(mismatch_pos_action_same)}")
        if len(mismatch_pos_action_same) > 0:
            cols = ["bar_index"]
            if "bar_time_ea" in m.columns:
                cols.append("bar_time_ea")
            cols += ["action_ea", "position_side_ea", "position_side_td"]
            print(mismatch_pos_action_same[cols].head(max_rows).to_string(index=False))


def report_feature_diff(m: pd.DataFrame, max_rows: int):
    feat_cols = [c for c in m.columns if c.startswith("feat_") and (c.endswith("_ea") or c.endswith("_td"))]
    if not feat_cols:
        return

    feat_ids = sorted({c.replace("_ea", "").replace("_td", "") for c in feat_cols})
    diffs = []
    for feat in feat_ids:
        col_a = f"{feat}_ea"
        col_b = f"{feat}_td"
        if col_a in m.columns and col_b in m.columns:
            d = (m[col_a] - m[col_b]).abs()
            d = d.dropna()
            if not d.empty:
                diffs.append(d.rename(feat))

    if not diffs:
        return

    diff_df = pd.concat(diffs, axis=1)
    mean_diff = diff_df.mean().sort_values(ascending=False)
    print("feature diff mean (top 5):")
    print(mean_diff.head(5).to_string())

    top_feat = mean_diff.index[0]
    col_a = f"{top_feat}_ea"
    col_b = f"{top_feat}_td"
    if col_a in m.columns and col_b in m.columns:
        top_rows = m.assign(_diff=(m[col_a] - m[col_b]).abs()).sort_values("_diff", ascending=False).head(max_rows)
        cols = ["bar_index"]
        if "bar_time_ea" in m.columns:
            cols.append("bar_time_ea")
        cols += [col_a, col_b, "_diff"]
        print(f"\nTop diffs for {top_feat}:")
        print(top_rows[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ea_log", default="ea_parity_log.csv")
    ap.add_argument("--td_log", default="td_parity_log.csv")
    ap.add_argument("--pip_decimal", type=float, default=0.0001)
    ap.add_argument("--max_rows", type=int, default=10)
    args = ap.parse_args()

    ea = read_parity_log(args.ea_log)
    td = read_parity_log(args.td_log)

    if "bar_index" in ea.columns and "bar_index" in td.columns:
        m = align_on_bar_index(ea, td)
    elif "bar_time" in ea.columns and "bar_time" in td.columns:
        m = ea.merge(td, on="bar_time", suffixes=("_ea", "_td"))
    else:
        raise RuntimeError("No common key to merge parity logs.")

    if len(m) == 0:
        raise RuntimeError("No overlapping rows after alignment.")

    print("Aligned rows:", len(m))

    if "position_side_ea" in m.columns and "position_side_td" in m.columns:
        mismatch = m[m["position_side_ea"] != m["position_side_td"]]
        print(f"position_side mismatches: {len(mismatch)}")
        if len(mismatch) > 0:
            cols = ["bar_index"]
            if "bar_time_ea" in m.columns:
                cols.append("bar_time_ea")
            cols += ["position_side_ea", "position_side_td"]
            print(mismatch[cols].head(args.max_rows).to_string(index=False))

    if "action_ea" in m.columns and "action_td" in m.columns:
        mismatch = m[m["action_ea"] != m["action_td"]]
        print(f"action mismatches: {len(mismatch)}")
        if len(mismatch) > 0:
            cols = ["bar_index"]
            if "bar_time_ea" in m.columns:
                cols.append("bar_time_ea")
            cols += ["action_ea", "action_td"]
            print(mismatch[cols].head(args.max_rows).to_string(index=False))

    report_price_diff(m, "ask", args.pip_decimal, args.max_rows)
    report_price_diff(m, "bid", args.pip_decimal, args.max_rows)
    report_price_diff(m, "entry_ask", args.pip_decimal, args.max_rows)
    report_price_diff(m, "entry_bid", args.pip_decimal, args.max_rows)

    report_numeric_diff(m, "pos_age_bars", args.max_rows)
    report_numeric_diff(m, "sl_pips", args.max_rows)
    report_numeric_diff(m, "lot", args.max_rows)
    report_numeric_diff(m, "exchange_rate", args.max_rows)
    report_conditional_diffs(m, args.pip_decimal, args.max_rows)
    report_mismatch_cross(m, args.max_rows)
    report_feature_diff(m, args.max_rows)


if __name__ == "__main__":
    main()
