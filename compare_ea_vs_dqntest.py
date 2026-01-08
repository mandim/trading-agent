import pandas as pd
import matplotlib.pyplot as plt

def read_csv_autosep(path: str) -> pd.DataFrame:
    """
    Robust CSV reader for either ';' or ',' delimited files.
    Tries python engine with automatic sep detection first; falls back to ';' then ','.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep=";")
        except Exception:
            return pd.read_csv(path, sep=",")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def parse_bar_time(series: pd.Series) -> pd.Series:
    """
    Parse bar_time formats:
    - EA: 'YYYY.MM.DD HH:MM'
    - Test: 'YYYY-MM-DDTHH:MM' or 'YYYY-MM-DD HH:MM' or same as EA
    """
    s = series.astype(str).str.strip()
    # Convert EA format to ISO-friendly:
    # '2025.01.03 00:00' -> '2025-01-03 00:00'
    s = s.str.replace(r"^(\d{4})\.(\d{2})\.(\d{2})", r"\1-\2-\3", regex=True)
    s = s.str.replace("T", " ", regex=False)

    return pd.to_datetime(s, errors="coerce")

def read_actions(path: str, label: str) -> pd.DataFrame:
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    # Candidate time columns
    time_col = find_column(df, ["bar_time", "bartime", "time", "datetime", "date"])
    if time_col is None:
        raise KeyError(f"[{label}] Could not find a bar_time-like column. Columns={list(df.columns)}")

    # Candidate action columns
    action_col = find_column(df, ["action", "action_id", "a"])
    if action_col is None:
        raise KeyError(f"[{label}] Could not find an action-like column. Columns={list(df.columns)}")

    bar_col = find_column(df, ["bar_index", "bar", "bar_idx"])

    cols = [time_col, action_col]
    if bar_col is not None:
        cols.append(bar_col)

    out = df[cols].copy()
    out = out.rename(columns={time_col: "bar_time", action_col: "action"})
    if bar_col is not None:
        out = out.rename(columns={bar_col: "bar_index"})
    out["bar_time"] = parse_bar_time(out["bar_time"])
    out["action"] = pd.to_numeric(out["action"], errors="coerce")
    if "bar_index" in out.columns:
        out["bar_index"] = pd.to_numeric(out["bar_index"], errors="coerce")

    out = out.dropna(subset=["bar_time", "action"]).copy()
    out["action"] = out["action"].astype(int)

    # If duplicates per bar_time exist, keep the last one (most recent decision)
    out = out.sort_values("bar_time").groupby("bar_time", as_index=False).tail(1)

    cols_out = ["bar_time", "action"]
    if "bar_index" in out.columns:
        cols_out.append("bar_index")
    return out[cols_out]

def read_mt4_trades(path: str) -> pd.DataFrame:
    """
    Reads the mt4_trades_2025.csv generated from StrategyTester.htm.
    Expected columns: Time, Profit (case-insensitive).
    """
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    time_col = find_column(df, ["time"])
    profit_col = find_column(df, ["profit"])

    if time_col is None or profit_col is None:
        raise KeyError(f"[MT4] Expected columns like Time/Profit. Columns={list(df.columns)}")

    t = df[[time_col, profit_col]].copy()
    t = t.rename(columns={time_col: "close_time", profit_col: "profit"})
    t["close_time"] = pd.to_datetime(t["close_time"], errors="coerce")
    t["profit"] = pd.to_numeric(t["profit"], errors="coerce")

    # In MT4 report, open rows often have NaN profit; keep realized rows
    t = t.dropna(subset=["close_time", "profit"]).copy()
    return t

def read_test_trades(path: str, actions_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Reads test_trades_2025.csv from test_dqn.
    We prefer profit_net_usd; fall back to profit_gross_usd.
    """
    df = read_csv_autosep(path)
    df = normalize_columns(df)

    close_col = find_column(df, ["close_time", "closetime"])
    if close_col is None:
        bar_col = find_column(df, ["bar_index", "bar", "bar_idx"])
        if bar_col is None or actions_df is None:
            raise KeyError(f"[Test trades] Missing close_time. Columns={list(df.columns)}")
        if ("bar_index" not in actions_df.columns) or ("bar_time" not in actions_df.columns):
            raise KeyError("[Test trades] Missing bar_index/bar_time in actions for close_time mapping.")

        tmp = actions_df[["bar_index", "bar_time"]].copy()
        tmp["bar_index"] = pd.to_numeric(tmp["bar_index"], errors="coerce")
        tmp = tmp.dropna(subset=["bar_index", "bar_time"]).drop_duplicates("bar_index")
        df["close_time"] = df[bar_col].map(tmp.set_index("bar_index")["bar_time"])
        close_col = "close_time"

    pnl_col = find_column(df, ["profit_net_usd", "profit_gross_usd", "profit", "pnl"])
    if pnl_col is None:
        raise KeyError(f"[Test trades] Missing pnl/profit column. Columns={list(df.columns)}")

    t = df[[close_col, pnl_col]].copy()
    t = t.rename(columns={close_col: "close_time", pnl_col: "profit"})
    t["close_time"] = pd.to_datetime(t["close_time"], errors="coerce")
    t["profit"] = pd.to_numeric(t["profit"], errors="coerce")
    t = t.dropna(subset=["close_time", "profit"]).copy()
    return t

def monthly_pnl(trades: pd.DataFrame) -> pd.Series:
    t = trades.copy()
    t["month"] = t["close_time"].dt.to_period("M").dt.to_timestamp()
    return t.groupby("month")["profit"].sum().sort_index()

def main():
    ea_actions_path = "ea_actions_2025.csv"
    td_actions_path = "test_actions_2025.csv"
    ea_trades_path  = "mt4_trades_2025.csv"
    td_trades_path  = "test_trades_2025.csv"

    ea = read_actions(ea_actions_path, "EA actions")
    td = read_actions(td_actions_path, "Test actions")

    if "bar_time" in ea.columns:
        print("[EA actions]  range:", ea["bar_time"].min(), "->", ea["bar_time"].max(), "rows=", len(ea))
    if "bar_time" in td.columns:
        print("[Test actions] range:", td["bar_time"].min(), "->", td["bar_time"].max(), "rows=", len(td))
    if "bar_index" in ea.columns:
        print("[EA actions]  bar_index:", ea["bar_index"].min(), "->", ea["bar_index"].max(), "rows=", len(ea))
    if "bar_index" in td.columns:
        print("[Test actions] bar_index:", td["bar_index"].min(), "->", td["bar_index"].max(), "rows=", len(td))

    merge_key = "bar_index" if ("bar_index" in ea.columns and "bar_index" in td.columns) else "bar_time"
    m = ea.merge(td, on=merge_key, suffixes=("_ea", "_td"))
    if len(m) == 0:
        # Diagnostics
        if "bar_time" in ea.columns:
            print("EA bar_time range:", ea["bar_time"].min(), "->", ea["bar_time"].max(), "rows=", len(ea))
        if "bar_time" in td.columns:
            print("TD bar_time range:", td["bar_time"].min(), "->", td["bar_time"].max(), "rows=", len(td))
        if "bar_index" in ea.columns:
            print("EA bar_index range:", ea["bar_index"].min(), "->", ea["bar_index"].max(), "rows=", len(ea))
        if "bar_index" in td.columns:
            print("TD bar_index range:", td["bar_index"].min(), "->", td["bar_index"].max(), "rows=", len(td))
        raise RuntimeError(f"No overlapping {merge_key} values. Check alignment and bar gating.")

    m["match"] = (m["action_ea"] == m["action_td"])
    match_rate = m["match"].mean()

    print(f"Aligned bars: {len(m)}")
    print(f"Action match rate: {match_rate*100:.2f}%")

    conf = pd.crosstab(m["action_ea"], m["action_td"], rownames=["EA"], colnames=["TestDQN"])
    print("\nConfusion matrix (counts):")
    print(conf)

    # Monthly PnL
    try:
        mt4_tr = read_mt4_trades(ea_trades_path)
        td_tr  = read_test_trades(td_trades_path, actions_df=td)

        mt4_m = monthly_pnl(mt4_tr)
        td_m  = monthly_pnl(td_tr)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(mt4_m.index, mt4_m.values, label="MT4 (EA)")
        ax.plot(td_m.index, td_m.values, label="TestDQN")
        ax.set_title("Monthly PnL (EA vs TestDQN)")
        ax.set_xlabel("Month")
        ax.set_ylabel("PnL (USD)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError as e:
        print(f"\nSkipping Monthly PnL plot (missing file): {e}")
    except KeyError as e:
        print(f"\nSkipping Monthly PnL plot (schema issue): {e}")

if __name__ == "__main__":
    main()
