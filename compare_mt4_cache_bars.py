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


def parse_bar_time(df: pd.DataFrame) -> pd.Series:
    dt_col = find_column(df, ["datetime", "timestamp"])
    if dt_col is not None:
        s = df[dt_col].astype(str).str.strip()
    else:
        date_col = find_column(df, ["date"])
        time_col = find_column(df, ["time"])
        if date_col is None and time_col is None:
            raise KeyError("Missing date/time columns.")
        if date_col is not None and time_col is not None:
            s = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        else:
            s = df[date_col or time_col].astype(str).str.strip()

    s = s.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    s = s.str.replace("T", " ", regex=False)
    return pd.to_datetime(s, errors="coerce")


def read_bars_csv(path: str) -> pd.DataFrame:
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    t = parse_bar_time(df)
    df["bar_time"] = t

    open_col = find_column(df, ["open", "o"])
    high_col = find_column(df, ["high", "h"])
    low_col = find_column(df, ["low", "l"])
    close_col = find_column(df, ["close", "c"])
    vol_col = find_column(df, ["volume", "vol", "tick_volume"])

    if any(c is None for c in [open_col, high_col, low_col, close_col]):
        raise KeyError(f"Missing OHLC columns in {path}. Columns={list(df.columns)}")

    out_cols = ["bar_time", open_col, high_col, low_col, close_col]
    if vol_col is not None:
        out_cols.append(vol_col)

    out = df[out_cols].copy()
    out = out.rename(
        columns={
            open_col: "open",
            high_col: "high",
            low_col: "low",
            close_col: "close",
            vol_col: "volume" if vol_col is not None else vol_col,
        }
    )

    out = out.dropna(subset=["bar_time", "close"])
    out["open"] = pd.to_numeric(out["open"], errors="coerce")
    out["high"] = pd.to_numeric(out["high"], errors="coerce")
    out["low"] = pd.to_numeric(out["low"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if "volume" in out.columns:
        out["volume"] = pd.to_numeric(out["volume"], errors="coerce")

    return out.dropna(subset=["open", "high", "low", "close"]).copy()


def read_cache(cache_dir: str) -> pd.DataFrame:
    bar_times = np.load(f"{cache_dir}/bar_times.npy")
    bars_features = np.load(f"{cache_dir}/bars_features.npy", mmap_mode="r")

    bar_times_dt = pd.to_datetime(bar_times, unit="s", errors="coerce")
    close = np.asarray(bars_features[:, 0], dtype=np.float64)
    return pd.DataFrame({"bar_time": bar_times_dt, "close": close})


def compare_two(a: pd.DataFrame, b: pd.DataFrame, label_a: str, label_b: str, max_rows: int = 10):
    merged = a.merge(b, on="bar_time", suffixes=(f"_{label_a}", f"_{label_b}"))
    print(f"[{label_a} vs {label_b}] rows: {len(merged)}")

    if len(merged) == 0:
        return

    ca = f"close_{label_a}"
    cb = f"close_{label_b}"
    merged["abs_diff"] = (merged[ca] - merged[cb]).abs()
    print(f"[{label_a} vs {label_b}] close abs diff: mean={merged['abs_diff'].mean():.10f} "
          f"max={merged['abs_diff'].max():.10f}")

    bad = merged[merged["abs_diff"] > 1e-6].sort_values("abs_diff", ascending=False)
    if len(bad) > 0:
        print(f"[{label_a} vs {label_b}] mismatched closes (>1e-6): {len(bad)}")
        print(bad[["bar_time", ca, cb, "abs_diff"]].head(max_rows).to_string(index=False))
    else:
        print(f"[{label_a} vs {label_b}] closes match within 1e-6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mt4_csv", required=True, help="MT4-exported bars CSV (Date/Time/Open/High/Low/Close)")
    ap.add_argument("--cache_dir", default="cache_fx_EURUSD_D1")
    ap.add_argument("--data_csv", default=None, help="Optional source bars CSV used in preprocessing")
    ap.add_argument("--time_offset_hours", type=float, default=0.0, help="Apply offset to MT4 times")
    ap.add_argument("--max_rows", type=int, default=10)
    args = ap.parse_args()

    mt4 = read_bars_csv(args.mt4_csv)
    if args.time_offset_hours != 0.0:
        mt4["bar_time"] = mt4["bar_time"] + pd.to_timedelta(args.time_offset_hours, unit="h")

    cache = read_cache(args.cache_dir)

    print("cache range:", cache["bar_time"].min(), "->", cache["bar_time"].max(), "rows=", len(cache))
    print("mt4 range:", mt4["bar_time"].min(), "->", mt4["bar_time"].max(), "rows=", len(mt4))

    cache_set = set(cache["bar_time"])
    mt4_set = set(mt4["bar_time"])
    print("mt4 missing in cache:", len(mt4_set - cache_set))
    print("cache missing in mt4:", len(cache_set - mt4_set))

    compare_two(cache, mt4, "cache", "mt4", max_rows=args.max_rows)

    if args.data_csv:
        data = read_bars_csv(args.data_csv)
        print("data range:", data["bar_time"].min(), "->", data["bar_time"].max(), "rows=", len(data))
        compare_two(data, mt4, "data", "mt4", max_rows=args.max_rows)
        compare_two(cache, data, "cache", "data", max_rows=args.max_rows)


if __name__ == "__main__":
    main()
