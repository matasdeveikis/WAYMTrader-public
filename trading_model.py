"""
Standard run:
    python trading_model.py --learner-start 2022-01-01 --learner-end 2023-01-01 --update-policy-fit-daily --hold-overnight 
"""
import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from typing import Any, List, Optional, Tuple, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, MonthLocator, date2num


def get_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def normalize_ticker_from_source(t: str) -> str:
    return t.replace(".", "-").strip().upper()

# Ticker aliases: maps canonical database tickers to their truncated/alternate forms found in the mentions CSV so lookups can resolve correctly.
TICKER_ALIASES: Dict[str, List[str]] = {
    "BRK-B": ["BRK"],
    "BF-B": ["BF"],
}


def resolve_ticker_to_canonical(conn, source_ticker: str) -> Optional[str]:
    """Resolve a ticker from the mentions file to its canonical database ticker.

    Checks sp500_companies first, then falls back to TICKER_ALIASES.
    """
    t = normalize_ticker_from_source(source_ticker)

    # Direct match in sp500_companies
    cur = conn.execute(
        "SELECT canonical_ticker FROM sp500_companies WHERE ticker=? LIMIT 1;",
        (t,),
    )
    r = cur.fetchone()
    if r:
        return r[0]

    # Check aliases: if the source ticker matches an alias value, return the key.
    for canonical, aliases in TICKER_ALIASES.items():
        if t in aliases:
            return canonical

    return None


def parse_date(value: str) -> Optional[str]:
    if pd.isna(value):
        return None
    value = str(value).replace("_", " ").strip()
    for fmt in ("%d %B %Y", "%Y-%m-%d", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except Exception:
            pass
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def load_mentions_matrix(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "trade_date"})
    long = df.melt(id_vars=["trade_date"], var_name="ticker", value_name="mentions")
    long["ticker"] = long["ticker"].astype(str).map(normalize_ticker_from_source)
    long["trade_date"] = long["trade_date"].map(parse_date)
    long = long.dropna(subset=["trade_date"])
    long["mentions"] = pd.to_numeric(long["mentions"], errors="coerce").fillna(0).astype(float)
    return long


def format_trade_date(value: Any) -> str:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"Invalid trade_date value: {value!r}")
    return dt.date().isoformat()


def get_open_close_for(conn, ticker: Any, trade_date: str) -> Optional[Tuple[float, float, str]]:
    t = normalize_ticker_from_source(str(ticker))
    cur = conn.execute(
        "SELECT tracker_ticker, open_price, close_price FROM sp500_tracker_prices WHERE tracker_ticker=? AND trade_date=? LIMIT 1;",
        (t, trade_date),
    )
    row = cur.fetchone()
    if row:
        return float(row[1]), float(row[2]), row[0]

    # Resolve via sp500_companies or TICKER_ALIASES
    canonical = resolve_ticker_to_canonical(conn, t)
    if canonical is None:
        canonical = t
    cur = conn.execute(
        "SELECT open_price, close_price FROM daily_prices WHERE ticker=? AND trade_date=? LIMIT 1;",
        (canonical, trade_date),
    )
    row = cur.fetchone()
    if row:
        return float(row[0]), float(row[1]), canonical
    return None


def compute_spy_benchmark(conn, dates: List[str], start_capital: float, tracker_ticker: str = "SPY") -> pd.DataFrame:
    shares = None
    benchmark_rows = []
    for d in dates:
        oc = get_open_close_for(conn, tracker_ticker, d)
        if oc is None:
            benchmark_rows.append({"trade_date": d, "spy_value": None})
            continue
        open_p, close_p, resolved = oc
        if shares is None:
            entry_price = open_p if open_p and open_p > 0 else close_p
            if entry_price is None or entry_price == 0:
                raise RuntimeError(f"Unable to determine SPY entry price for benchmark on {d}")
            shares = start_capital / entry_price
        benchmark_rows.append({"trade_date": d, "spy_value": shares * close_p if close_p is not None else None})
    return pd.DataFrame(benchmark_rows)


def build_features(mentions_df: pd.DataFrame, learner_start: str, end: Optional[str]) -> pd.DataFrame:
    # Filter to only include data from learner_start onward BEFORE computing expanding stats.
    # This prevents early mention data from contaminating z-scores used in training and trading.
    df = mentions_df[mentions_df["trade_date"] >= learner_start].copy()
    if end:
        df = df[df["trade_date"] <= end]
    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["prev_mentions"] = df.groupby("ticker")["mentions"].shift(1).fillna(0.0)
    df["mentions"] = pd.to_numeric(df["mentions"], errors="coerce").fillna(0.0).astype(float)
    df["mentions_momentum"] = df["mentions"] - df["prev_mentions"]
    df["mentions_rank"] = df.groupby("trade_date")["mentions"].rank(method="dense", pct=True).fillna(0.0)

    # Compute ticker baselines from the historical mentions series up to each row so
    # the daily refit uses the same feature definition regardless of learner_end.
    # Use expanding then shift so current row does not include its own value.
    df["mean_mentions"] = (
        df.groupby("ticker")["mentions"].transform(lambda s: s.expanding().mean().shift(1))
    )
    df["std_mentions"] = (
        df.groupby("ticker")["mentions"].transform(lambda s: s.expanding().std(ddof=0).shift(1))
    )
    # Coerce baseline columns to numeric floats to avoid static typing and operator issues
    df["mean_mentions"] = pd.to_numeric(df["mean_mentions"], errors="coerce").fillna(0.0).astype(float)
    df["std_mentions"] = pd.to_numeric(df["std_mentions"], errors="coerce").replace(0, np.nan).fillna(1.0).astype(float)
    df["z_score"] = (df["mentions"] - df["mean_mentions"]) / df["std_mentions"]
    df["z_score"] = pd.to_numeric(df["z_score"], errors="coerce").fillna(0.0).astype(float)
    return df


def merge_prices(conn, feature_df: pd.DataFrame, hold_overnight: bool = False) -> pd.DataFrame:
    rows = []
    for row in feature_df.itertuples(index=False):
        date_str = format_trade_date(row.trade_date)
        oc = get_open_close_for(conn, row.ticker, date_str)
        if oc is None:
            continue
        open_p, close_p, resolved = oc
        if open_p is None or close_p is None:
            continue
        if open_p == 0:
            continue
        ret = (close_p - open_p) / open_p
        rows.append({
            "trade_date": date_str,
            "ticker": row.ticker,
            "mentions": row.mentions,
            "z_score": row.z_score,
            "mentions_momentum": row.mentions_momentum,
            "mentions_rank": row.mentions_rank,
            "mean_mentions": row.mean_mentions,
            "std_mentions": row.std_mentions,
            "open": open_p,
            "close": close_p,
            "return_pct": ret,
        })
    df = pd.DataFrame(rows)
    if hold_overnight and not df.empty:
        df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
        df["next_open"] = df.groupby("ticker")["open"].shift(-1)
        # Compute overnight return: (next_open - open) / open
        mask = df["next_open"].notna() & (df["next_open"] > 0)
        df.loc[mask, "return_pct"] = (df.loc[mask, "next_open"] - df.loc[mask, "open"]) / df.loc[mask, "open"]
        df = df.drop(columns=["next_open"])
    return df


def fit_linear_policy(training_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    X = training_df[features].fillna(0.0).to_numpy(dtype=float)
    y = training_df["return_pct"].to_numpy(dtype=float)
    if X.shape[0] == 0:
        raise RuntimeError("No learner rows available to fit the linear policy.")
    X_design = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    weights, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    return weights


def predict_scores(weights: np.ndarray, feature_df: pd.DataFrame, features: List[str]) -> np.ndarray:
    X = feature_df[features].fillna(0.0).to_numpy(dtype=float)
    intercept = weights[0]
    coefs = weights[1:]
    return intercept + X.dot(coefs)


def save_policy_weights(weights: np.ndarray, feature_names: List[str], output_path: str) -> None:
    rows = []
    rows.append({"feature": "intercept", "weight": float(weights[0])})
    for name, coef in zip(feature_names, weights[1:]):
        rows.append({"feature": name, "weight": float(coef)})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def build_refit_training_df(price_df: pd.DataFrame, current_date: Any, feature_cols: List[str], strict: bool = False) -> pd.DataFrame:
    current_dt = pd.to_datetime(current_date)
    mask = price_df["trade_date"] < current_dt if strict else price_df["trade_date"] <= current_dt
    subset = price_df.loc[mask, feature_cols + ["return_pct"]]
    return pd.DataFrame(subset).copy()


def run_backtest(
    mentions_df: pd.DataFrame,
    db_path: str,
    learner_start: str,
    end: Optional[str],
    learner_end: str,
    top_k: int,
    z_threshold: Optional[float],
    min_mentions: int,
    position_cap: float,
    start_capital: float,
    transaction_cost: float,
    buy_spy_overnight: bool,
    hold_overnight: bool,
    update_policy_fit_daily: bool,
    allow_shorting: bool,
    position_cap_short: float = 0.1,
    rolling_window: int = 90,
    walk_forward_date: Optional[str] = None,
    slim: bool = False,
    extra_plots: bool = False,
    disable_top_mentions: bool = False,
):
    os.makedirs("results", exist_ok=True)
    conn = get_connection(db_path)
    feature_df = build_features(mentions_df, learner_start, end)
    if feature_df.empty:
        raise RuntimeError("No mention data available after applying start/end filters.")
    price_df = merge_prices(conn, feature_df, hold_overnight=hold_overnight)
    if price_df.empty:
        raise RuntimeError("No valid price-matched rows available for the selected date range.")
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    learner_end_dt = pd.to_datetime(learner_end)
    # Trading days on/after the walk-forward date are treated as an out-of-sample
    # walk-forward test (policy re-fit daily on strictly prior data, no lookahead).
    walk_forward_dt = pd.to_datetime(walk_forward_date) if walk_forward_date else None

    # Build the full trading calendar from the database so days without mention data are still processed.
    all_db_dates: List[str] = []
    end_dt = pd.to_datetime(end) if end else pd.Timestamp.max
    cur = conn.execute(
        "SELECT DISTINCT trade_date FROM daily_prices WHERE trade_date > ? AND trade_date <= ? ORDER BY 1;",
        (learner_end, str(end_dt)),
    )
    for (d,) in cur.fetchall():
        all_db_dates.append(pd.to_datetime(d).date().isoformat())

    # For each DB date, ensure there is at least one row so the groupby loop always runs.
    cal_rows = []
    for d in all_db_dates:
        oc = get_open_close_for(conn, "SPY", d)
        if oc is not None:
            open_p, close_p, _ = oc
            ret = (close_p - open_p) / open_p if open_p and open_p > 0 else 0.0
            cal_rows.append({"trade_date": pd.to_datetime(d), "ticker": "SPY", "mentions": 0.0, "z_score": 0.0,
                             "mentions_momentum": 0.0, "mentions_rank": 0.0, "mean_mentions": 0.0, "std_mentions": 1.0,
                             "open": open_p, "close": close_p, "return_pct": ret})
    cal_df = pd.DataFrame(cal_rows) if cal_rows else pd.DataFrame()

    # Merge: keep all mention-based rows (price_df), and add placeholder SPY rows for DB dates not already covered.
    existing_dates = set(price_df["trade_date"].dt.date.astype(str))
    extra_rows = []
    for _, row in cal_df.iterrows():
        d_str = row["trade_date"].date().isoformat()
        if d_str not in existing_dates:
            extra_rows.append(row)
    if extra_rows:
        price_df = pd.concat([price_df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Trading-period dates that have mention data but no price data yet (e.g. the latest
    # day before the market has printed prices) are added as zero-price placeholder rows so
    # the trading loop still generates trades. Price-derived trade fields are zeroed below.
    mention_dates = set(feature_df["trade_date"].dt.date.astype(str))
    covered_dates = set(price_df["trade_date"].dt.date.astype(str))
    # Only dates at/after the price-data frontier are treated as pending (e.g. the latest
    # trading day whose prices have not been published yet). Historical market-holiday gaps
    # that fall before later priced trading days are intentionally excluded.
    pending_cutoff = max(covered_dates) if covered_dates else learner_end
    missing_dates = sorted(
        d for d in mention_dates
        if d not in covered_dates and d > learner_end and d >= pending_cutoff
    )
    missing_feature_rows = []
    if missing_dates:
        missing_feat = feature_df[feature_df["trade_date"].dt.date.astype(str).isin(missing_dates)]
        for mrow in missing_feat.itertuples(index=False):
            missing_feature_rows.append({
                "trade_date": mrow.trade_date,
                "ticker": mrow.ticker,
                "mentions": mrow.mentions,
                "z_score": mrow.z_score,
                "mentions_momentum": mrow.mentions_momentum,
                "mentions_rank": mrow.mentions_rank,
                "mean_mentions": mrow.mean_mentions,
                "std_mentions": mrow.std_mentions,
                "open": 0.0,
                "close": 0.0,
                "return_pct": 0.0,
            })
    if missing_feature_rows:
        price_df = pd.concat([price_df, pd.DataFrame(missing_feature_rows)], ignore_index=True)

    # Now split into learner vs trading periods using the full set of dates.
    learner_end_dt = pd.to_datetime(learner_end)
    learner_mask = price_df["trade_date"] <= learner_end_dt
    trade_mask = price_df["trade_date"] > learner_end_dt
    learner_df = price_df[learner_mask].copy()
    trading_df = price_df[trade_mask].copy()
    diagnostics = {
        "learner_dates": int(learner_df["trade_date"].nunique()),
        "trading_dates": int(trading_df["trade_date"].nunique()),
        "z_threshold_enabled": bool(z_threshold is not None),
        "top_k": int(top_k),
        "min_mentions": int(min_mentions),
        "position_cap": float(position_cap),
        "transaction_cost": float(transaction_cost),
        "buy_spy_overnight": bool(buy_spy_overnight),
        "hold_overnight": bool(hold_overnight),
        "update_policy_fit_daily": bool(update_policy_fit_daily),
        "allow_shorting": bool(allow_shorting),
        "rolling_window": int(rolling_window),
        "walk_forward_date": walk_forward_date,
    }
    feature_cols = ["mentions", "z_score", "mentions_momentum", "mentions_rank"]
    training_df = learner_df[feature_cols + ["return_pct"]].copy()
    model_weights = fit_linear_policy(training_df, feature_cols)
    save_policy_weights(model_weights, feature_cols, "results/policy_weights.csv")
    training_history = []
    for name, weight in zip(["intercept"] + feature_cols, model_weights):
        training_history.append({"event": "learner_fit", "trade_date": None, "feature": name, "weight": float(weight)})
    daily_rows = []
    trades = []
    total_positions = []
    portfolio_value = float(start_capital)
    zero_candidate_dates: List[str] = []  # dates where no rows passed min_mentions/z_threshold filters
    zero_tradable_dates: List[str] = []   # dates where candidates existed but none were tradable (score filter)

    def apply_spy_overnight(current_date, next_date, value: float) -> Tuple[float, float, float, float]:
        if not buy_spy_overnight or next_date is None or value <= 0:
            return value, 0.0, 0.0, 0.0
        current_spy = get_open_close_for(conn, "SPY", format_trade_date(current_date))
        next_spy = get_open_close_for(conn, "SPY", format_trade_date(next_date))
        if current_spy is None or next_spy is None:
            return value, 0.0, 0.0, 0.0
        _, current_close, _ = current_spy
        next_open, _, _ = next_spy
        if not current_close or not next_open:
            return value, 0.0, 0.0, 0.0
        overnight_return = (next_open - current_close) / current_close
        pnl = value * overnight_return
        cost = value * transaction_cost * 2.0
        return value + pnl - cost, overnight_return, pnl - cost, cost

    def get_trade_exit(trade_row: Any, current_date: Any, next_date: Any) -> Tuple[str, float, str]:
        if hold_overnight and next_date is not None:
            next_prices = get_open_close_for(conn, trade_row.ticker, format_trade_date(next_date))
            if next_prices is not None:
                next_open, _, _ = next_prices
                if next_open:
                    return format_trade_date(next_date), float(next_open), "next_open"
        return format_trade_date(current_date), float(trade_row.close), "same_day_close"

    trading_groups = list(trading_df.groupby("trade_date"))
    for index, (date, day_df) in enumerate(trading_groups):
        next_date = trading_groups[index + 1][0] if index + 1 < len(trading_groups) else None
        # Days on/after the walk-forward date are out-of-sample walk-forward test days.
        # ISO date strings compare correctly lexicographically, so no datetime conversion needed.
        in_walk_forward = bool(walk_forward_date is not None and format_trade_date(date) >= walk_forward_date)
        candidate = day_df.copy()
        candidate = candidate[candidate["mentions"] >= min_mentions]
        if z_threshold is not None:
            candidate = candidate[candidate["z_score"] >= z_threshold]
        if candidate.empty:
            zero_candidate_dates.append(format_trade_date(date))
            portfolio_value_end, overnight_return, overnight_pnl, overnight_cost = apply_spy_overnight(date, next_date, portfolio_value)
            daily_rows.append({
                "trade_date": format_trade_date(date),
                "walk_forward": bool(in_walk_forward),
                "n_candidates": 0,
                "n_selected": 0,
                "start_value": portfolio_value,
                "end_value": portfolio_value_end,
                "pnl": overnight_pnl,
                "intraday_pnl": 0.0,
                "overnight_spy_return": overnight_return,
                "overnight_spy_pnl": overnight_pnl,
                "overnight_spy_cost": overnight_cost,
                "total_costs": overnight_cost,
            })
            portfolio_value = portfolio_value_end
            continue

        candidate["score"] = predict_scores(model_weights, candidate, feature_cols)
        candidate = candidate.sort_values("score", ascending=False)

        if top_k and len(candidate) > top_k:
            candidate = candidate.head(top_k)

        # Build tradable set based on shorting mode.
        if allow_shorting:
            tradable = candidate[candidate["score"] != 0].copy()
            score_weights = np.abs(tradable["score"].to_numpy(dtype=float))
        else:
            tradable = candidate[candidate["score"] > 0].copy()
            score_weights = tradable["score"].to_numpy(dtype=float)

        # Optionally drop the day's top-mention ticker so it is never traded that day.
        if disable_top_mentions and not day_df.empty:
            top_mention_ticker = day_df.loc[day_df["mentions"].idxmax(), "ticker"]
            tradable = tradable[tradable["ticker"] != top_mention_ticker].copy()
            if allow_shorting:
                score_weights = np.abs(tradable["score"].to_numpy(dtype=float))
            else:
                score_weights = tradable["score"].to_numpy(dtype=float)

        if in_walk_forward or update_policy_fit_daily:
            # Walk-forward days always re-fit the policy using strictly prior data
            # (no lookahead), so the walk-forward portion is a genuine out-of-sample test.
            refit_df = build_refit_training_df(price_df, date, feature_cols, strict=in_walk_forward)
            if not refit_df.empty:
                model_weights = fit_linear_policy(refit_df, feature_cols)
                save_policy_weights(model_weights, feature_cols, "results/policy_weights.csv")
                event = "walk_forward_refit" if in_walk_forward else "daily_refit"
                for name, weight in zip(["intercept"] + feature_cols, model_weights):
                    training_history.append({"event": event, "trade_date": format_trade_date(date), "feature": name, "weight": float(weight)})

        if tradable.empty or score_weights.sum() <= 0:
            portfolio_value_end, overnight_return, overnight_pnl, overnight_cost = apply_spy_overnight(date, next_date, portfolio_value)
            daily_rows.append({
                "trade_date": format_trade_date(date),
                "walk_forward": bool(in_walk_forward),
                "n_candidates": int(len(day_df)),
                "n_selected": 0,
                "start_value": portfolio_value,
                "end_value": portfolio_value_end,
                "pnl": overnight_pnl,
                "intraday_pnl": 0.0,
                "overnight_spy_return": overnight_return,
                "overnight_spy_pnl": overnight_pnl,
                "overnight_spy_cost": overnight_cost,
                "total_costs": overnight_cost,
            })
            portfolio_value = portfolio_value_end
            continue

        allocations = []
        weight_sum = score_weights.sum()
        # Compute raw proportional allocations, then redistribute clipped amounts so the full budget is used.
        n = len(score_weights)
        caps = np.array([position_cap_short if float(pd.to_numeric(trade_row.score, errors="coerce")) < 0 else position_cap for trade_row in tradable.itertuples(index=False)])
        allocs = portfolio_value * score_weights / weight_sum
        capped = set()
        for _ in range(n):
            clipped_mask = np.array([i not in capped and allocs[i] > portfolio_value * caps[i] for i in range(n)])
            if not clipped_mask.any():
                break
            excess = (allocs[clipped_mask] - portfolio_value * caps[clipped_mask]).sum()
            uncapped_sum = score_weights[[i for i in range(n) if i not in capped]].sum()
            if uncapped_sum <= 0:
                break
            allocs[clipped_mask] = portfolio_value * caps[clipped_mask]
            uncapped_idx = [i for i in range(n) if i not in capped and not clipped_mask[i]]
            for i in uncapped_idx:
                allocs[i] += excess * score_weights[i] / uncapped_sum
            capped.update(i for i in range(n) if clipped_mask[i])
        allocations = allocs.tolist()

        day_pnl = 0.0
        total_costs = 0.0
        for alloc, trade_row in zip(allocations, tradable.itertuples(index=False)):
            open_price = float(pd.to_numeric(trade_row.open, errors="coerce"))
            close_price = float(pd.to_numeric(trade_row.close, errors="coerce"))
            score_val = float(pd.to_numeric(trade_row.score, errors="coerce"))
            is_short = score_val < 0
            side = "short" if is_short else "long"

            if not open_price:
                # No price data for this day (e.g. the latest day before the market has
                # printed prices): keep the trade with price-derived fields zeroed as
                # placeholders so trades.csv still records everything calculable from mentions.
                exit_date = format_trade_date(date)
                exit_price = 0.0
                exit_timing = "0"
                ret = 0.0
                pnl = 0.0
                cost = 0.0
            else:
                exit_date, exit_price, exit_timing = get_trade_exit(trade_row, date, next_date)
                exit_price = float(pd.to_numeric(exit_price, errors="coerce"))
                if is_short:
                    ret = (open_price - exit_price) / open_price
                else:
                    ret = (exit_price - open_price) / open_price
                pnl = alloc * ret
                cost = alloc * transaction_cost * 2.0
                day_pnl += pnl - cost
                total_costs += cost

            trades.append({
                "trade_date": format_trade_date(date),
                "exit_date": exit_date,
                "ticker": trade_row.ticker,
                "side": side,
                "alloc": alloc,
                "open": open_price,
                "close": close_price,
                "exit_price": exit_price,
                "exit_timing": exit_timing,
                "return": ret,
                "pnl": pnl,
                "cost": cost,
                "mentions": trade_row.mentions,
                "z_score": trade_row.z_score,
                "score": trade_row.score,
            })

        portfolio_value_after_trades = portfolio_value + day_pnl
        portfolio_value_end, overnight_return, overnight_pnl, overnight_cost = apply_spy_overnight(date, next_date, portfolio_value_after_trades)
        daily_rows.append({
            "trade_date": format_trade_date(date),
            "walk_forward": bool(in_walk_forward),
            "n_candidates": int(len(day_df)),
            "n_selected": int(len(tradable)),
            "start_value": portfolio_value,
            "end_value": portfolio_value_end,
            "pnl": day_pnl + overnight_pnl,
            "intraday_pnl": day_pnl,
            "overnight_spy_return": overnight_return,
            "overnight_spy_pnl": overnight_pnl,
            "overnight_spy_cost": overnight_cost,
            "total_costs": total_costs + overnight_cost,
        })
        portfolio_value = portfolio_value_end
        total_positions.append(len(tradable))

    # Build DataFrames with explicit column order so empty trading periods still
    # produce schema-correct (header-only) output files instead of column-less frames.
    daily_df = pd.DataFrame(daily_rows, columns=[
        "trade_date", "walk_forward", "n_candidates", "n_selected", "start_value",
        "end_value", "pnl", "intraday_pnl", "overnight_spy_return", "overnight_spy_pnl",
        "overnight_spy_cost", "total_costs",
    ])
    trades_df = pd.DataFrame(trades, columns=[
        "trade_date", "exit_date", "ticker", "side", "alloc", "open", "close",
        "exit_price", "exit_timing", "return", "pnl", "cost", "mentions", "z_score", "score",
    ])
    diagnostics["average_positions_per_day"] = float(np.mean(total_positions)) if total_positions else 0.0
    pd.DataFrame([diagnostics]).to_csv("results/strategy_diagnostics.csv", index=False)
    pd.DataFrame(training_history).to_csv("results/training_history.csv", index=False)
    daily_df.to_csv("results/backtest_daily.csv", index=False)
    trades_df.to_csv("results/trades.csv", index=False)

    # Write debug file for trade dates where n_candidates was 0 (only if any exist).
    os.makedirs("results", exist_ok=True)
    if zero_candidate_dates:
        pd.DataFrame({"trade_date": sorted(zero_candidate_dates)}).to_csv(
            "results/zero_n_candidates.csv", index=False,
        )

    ticker_stats_cols = [
        "ticker",
        "n_trades",
        "total_pnl",
        "total_costs",
        "avg_return",
        "avg_alloc",
        "total_alloc",
        "mean_z_score",
        "mean_mentions",
        "win_rate",
    ]
    if trades_df.empty:
        ticker_stats = pd.DataFrame(columns=ticker_stats_cols)
    else:
        ticker_stats = trades_df[trades_df["return"].notna()].groupby("ticker", as_index=False).agg(
            n_trades=("trade_date", "count"),
            total_pnl=("pnl", "sum"),
            total_costs=("cost", "sum"),
            avg_return=("return", "mean"),
            avg_alloc=("alloc", "mean"),
            total_alloc=("alloc", "sum"),
            mean_z_score=("z_score", "mean"),
            mean_mentions=("mentions", "mean"),
            win_rate=("pnl", lambda s: float((s > 0).sum()) / len(s) if len(s) > 0 else float("nan")),
        )
        ticker_stats = ticker_stats[ticker_stats_cols]
        ticker_stats = ticker_stats.sort_values("total_pnl", ascending=False)
    ticker_stats.to_csv("results/ticker_stats.csv", index=False)

    if trading_df.empty:
        # No trading days after learner_end (e.g. learner-end = previous trading day
        # with no newer data yet). Emit an empty, schema-correct plot frame so the
        # metric/plotting code below is skipped and the result CSVs stay empty.
        plot_df = pd.DataFrame(columns=["trade_date", "start_value", "end_value", "spy_value"])
    else:
        benchmark_dates = sorted(trading_df["trade_date"].dt.date.astype(str).unique().tolist())
        benchmark_df = compute_spy_benchmark(conn, benchmark_dates, start_capital)
        plot_df = daily_df.merge(benchmark_df, on="trade_date", how="left").dropna(subset=["spy_value"]).copy()
    conn.close()
    # Defaults for CAPM / correlation outputs so they are always bound when referenced below.
    x = np.array([])
    y = np.array([])
    capm_alpha_daily = capm_alpha_annual = capm_beta = capm_r2 = capm_cov = float("nan")
    # Defaults for performance metrics so they are always bound for plot titles and prints.
    value_df = pd.DataFrame()
    first_row = pd.Series(dtype=float)
    final_row = pd.Series(dtype=float)
    years = float("nan")
    strategy_yoy_gain = float("nan")
    strategy_sharpe = float("nan")
    strategy_sortino = float("nan")
    total_trades = 0
    max_drawdown_pct = 0.0
    if not plot_df.empty:
        plot_df["trade_date"] = pd.to_datetime(plot_df["trade_date"])
        plot_df = plot_df.sort_values("trade_date")
        plot_df["cum_high"] = plot_df["end_value"].cummax()
        plot_df["drawdown_pct"] = (plot_df["end_value"] / plot_df["cum_high"] - 1.0) * 100.0
        plot_df["year"] = plot_df["trade_date"].dt.year
        annual_returns = plot_df.groupby("year", as_index=False).agg(
            start_date=("trade_date", "first"),
            start_value=("start_value", "first"),
            end_date=("trade_date", "last"),
            end_value=("end_value", "last"),
        )
        annual_returns["days"] = (annual_returns["end_date"] - annual_returns["start_date"]).dt.days.replace(0, 1)
        annual_returns["annual_return_pct"] = np.where(
            annual_returns["start_value"] > 0,
            (annual_returns["end_value"] / annual_returns["start_value"]) ** (365.25 / annual_returns["days"]) - 1.0,
            0.0,
        ) * 100.0

        if not trades_df.empty:
            trades_df["trade_date"] = pd.to_datetime(trades_df["trade_date"])
            trades_df["trade_date"] = trades_df["trade_date"].dt.normalize()
            trades_df["day"] = trades_df["trade_date"]
            trades_per_day = trades_df.groupby("day").size().reset_index(name="trades")
        else:
            trades_per_day = pd.DataFrame(columns=["day", "trades"])

        # 30-day rolling correlation of strategy vs SPY daily returns to assess
        # how dependent the strategy's performance is on the market over time.
        corr_df = plot_df[["trade_date", "end_value", "spy_value"]].copy()
        corr_df["strat_ret"] = corr_df["end_value"].pct_change()
        corr_df["spy_ret"] = corr_df["spy_value"].pct_change()
        corr_df["rolling_corr"] = corr_df["strat_ret"].rolling(30).corr(corr_df["spy_ret"])

        # CAPM regression analysis: regress the strategy's daily returns on SPY
        # daily returns to isolate alpha (intercept) and beta (slope) vs the market.
        capm_df = corr_df.dropna(subset=["strat_ret", "spy_ret"]).copy()
        if len(capm_df) > 2:
            x = capm_df["spy_ret"].to_numpy(dtype=float)
            y = capm_df["strat_ret"].to_numpy(dtype=float)
            design = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            capm_alpha_daily = float(coef[0])
            capm_beta = float(coef[1])
            residuals = y - design.dot(coef)
            ss_res = float(np.sum(residuals ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            capm_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            capm_cov = float(np.cov(y, x)[0, 1])
            capm_alpha_annual = (1.0 + capm_alpha_daily) ** 252 - 1.0
        else:
            capm_alpha_daily = capm_beta = capm_r2 = capm_alpha_annual = float("nan")

        # Pre-compute performance metrics used in the plot titles and the summary printout.
        value_df = plot_df.sort_values("trade_date")
        first_row = value_df.iloc[0]
        final_row = value_df.iloc[-1]
        elapsed_days = max((final_row["trade_date"] - first_row["trade_date"]).days, 1)
        years = elapsed_days / 365.25
        strategy_yoy_gain = (float(final_row["end_value"]) / start_capital) ** (1.0 / years) - 1.0
        strategy_returns = value_df["end_value"] / value_df["start_value"] - 1.0
        strategy_return_std = strategy_returns.std(ddof=1)
        strategy_sharpe = (
            strategy_returns.mean() / strategy_return_std * np.sqrt(252)
            if strategy_return_std and not pd.isna(strategy_return_std)
            else float("nan")
        )
        # Downside deviation for Sortino ratio (only negative returns).
        downside_returns = strategy_returns[strategy_returns < 0]
        strategy_downside_std = downside_returns.std(ddof=1) if len(downside_returns) > 1 else float("nan")
        strategy_sortino = (
            strategy_returns.mean() / strategy_downside_std * np.sqrt(252)
            if not pd.isna(strategy_downside_std) and strategy_downside_std != 0
            else float("nan")
        )
        total_trades = int(len(trades_df))
        max_drawdown_pct = float(plot_df["drawdown_pct"].min()) if not plot_df["drawdown_pct"].empty else 0.0

        if not slim:
            # The PnL distribution and Score-vs-Win-Rate panels are only rendered when
            # --extra-plots is requested; drop the last grid row otherwise.
            n_grid_rows = 6 if extra_plots else 5
            grid_heights = [1.4, 1, 1, 1, 1, 1] if extra_plots else [1.4, 1, 1, 1, 1]
            fig = plt.figure(figsize=(16, 28))
            gs = fig.add_gridspec(nrows=n_grid_rows, ncols=2, height_ratios=grid_heights, hspace=0.40, wspace=0.15)
            ax_main = fig.add_subplot(gs[0:2, :])
            ax_drawdown = fig.add_subplot(gs[2, 0])
            ax_annual = fig.add_subplot(gs[2, 1])
            ax_trades_day = fig.add_subplot(gs[3, 0])
            ax_pnl_rate = fig.add_subplot(gs[3, 1])
            # Bottom two rows reordered: CAPM + rolling correlation, then PnL + score-vs-winrate.
            ax_capm = fig.add_subplot(gs[4, 0])
            ax_corr = fig.add_subplot(gs[4, 1])

            ax_main.plot(plot_df["trade_date"], plot_df["end_value"], label="Strategy", color="#1f77b4")
            ax_main.plot(plot_df["trade_date"], plot_df["spy_value"], label="SPY benchmark", color="#ff7f0e")
            if walk_forward_dt is not None:
                ax_main.axvline(x=float(date2num(walk_forward_dt)), color="orange", linestyle="--", linewidth=1.6,
                                label="Walk-forward start")
            ax_main.set_xlabel("trade_date")
            ax_main.set_ylabel("Portfolio value")
            sharpe_str = f"{strategy_sharpe:.2f}" if not pd.isna(strategy_sharpe) else "N/A"
            sortino_str = f"{strategy_sortino:.2f}" if not pd.isna(strategy_sortino) else "N/A"
            ax_main.set_title(
                f"Equity Curve\n"
                f"Sharpe: {sharpe_str} | Sortino: {sortino_str} | Trades: {total_trades:,} | CAGR: {strategy_yoy_gain:.2%}"
            )
            ax_main.legend()
            ax_main.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
            ax_main.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            ax_main.tick_params(axis="x", rotation=45)

            ax_drawdown.bar(plot_df["trade_date"], plot_df["drawdown_pct"], color="#d62728")
            if walk_forward_dt is not None:
                ax_drawdown.axvline(x=float(date2num(walk_forward_dt)), color="orange", linestyle="--", linewidth=1.2)
            ax_drawdown.set_title(f"Daily drawdown (%) | Max: {max_drawdown_pct:.2f}%")
            ax_drawdown.set_ylabel("Drawdown %")
            ax_drawdown.set_xlabel("Day")
            ax_drawdown.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
            ax_drawdown.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            ax_drawdown.tick_params(axis="x", rotation=45)

            ax_annual.bar(annual_returns["year"].astype(str), annual_returns["annual_return_pct"], color="#2ca02c", edgecolor="black")
            ax_annual.set_title("Annual return (%)")
            ax_annual.set_xlabel("Year")
            ax_annual.set_ylabel("Return %")
            for idx, value in enumerate(annual_returns["annual_return_pct"]):
                ax_annual.text(idx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
            # When the walk-forward starts on Jan 1 of a year, draw a separation between
            # the last backtested year's bar and the first walk-forward year's bar.
            if walk_forward_dt is not None and walk_forward_dt.month == 1 and walk_forward_dt.day == 1:
                year_vals = annual_returns["year"].tolist()
                if walk_forward_dt.year in year_vals:
                    wf_idx = year_vals.index(walk_forward_dt.year)
                    if wf_idx > 0:
                        ax_annual.axvline(x=wf_idx - 0.5, color="orange", linestyle="--", linewidth=1.2)

            if not trades_per_day.empty:
                # With hundreds of daily bars spread over a multi-year axis, the default
                # 0.8-day bar width is sub-pixel (< 1 px) and renders as a dense, unreadable
                # band, which makes the chart look like trades are missing. Widen the bars
                # so every trading day with trades is clearly visible.
                ax_trades_day.bar(trades_per_day["day"].dt.to_pydatetime(), trades_per_day["trades"],
                                  width=1.0, align="center", color="#9467bd")
                if walk_forward_dt is not None:
                    ax_trades_day.axvline(x=float(date2num(walk_forward_dt)), color="orange", linestyle="--", linewidth=1.2)
                avg_trades_per_day = float(trades_per_day["trades"].mean())
                ax_trades_day.set_title(f"Trades per day | Avg: {avg_trades_per_day:.1f}/day")
                ax_trades_day.set_xlabel("Day")
                ax_trades_day.set_ylabel("Number of trades")
                ax_trades_day.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
                ax_trades_day.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
                ax_trades_day.tick_params(axis="x", rotation=45)
            else:
                ax_trades_day.text(0.5, 0.5, "No trades available", ha="center", va="center")
                ax_trades_day.set_axis_off()

            # Compute N-day rolling win rate from individual trades.
            # For each trading day, the window is (day - rolling_window days; day].
            # Win rate = wins in window / total trades in window.
            if not trades_df.empty:
                trades_plot = trades_df[["trade_date", "pnl"]].dropna().copy()
                trades_plot["trade_date"] = pd.to_datetime(trades_plot["trade_date"])
                trades_plot["is_win"] = trades_plot["pnl"] > 0
                # For each unique trade date, compute the rolling win rate.
                all_trade_dates = sorted(trades_plot["trade_date"].unique())
                win_rates = []
                for d in all_trade_dates:
                    window_start = d - pd.Timedelta(days=rolling_window)
                    mask = (trades_plot["trade_date"] > window_start) & (trades_plot["trade_date"] <= d)
                    subset = trades_plot.loc[mask]
                    if len(subset) == 0:
                        win_rates.append(0.0)
                    else:
                        win_rates.append(subset["is_win"].sum() / len(subset) * 100.0)
                daily_win = pd.DataFrame({"trade_date": all_trade_dates, "win_rate_30d": win_rates})

                start_dt = learner_end_dt + pd.Timedelta(days=rolling_window)
                daily_win = daily_win[daily_win["trade_date"] >= start_dt].copy()
                if not daily_win.empty:
                    ax_pnl_rate.plot(
                        daily_win["trade_date"],
                        daily_win["win_rate_30d"],
                        color="#2ca02c",
                        linewidth=1.5,
                    )
                    ax_pnl_rate.fill_between(daily_win["trade_date"], daily_win["win_rate_30d"], alpha=0.25, color="#2ca02c")
                    ax_pnl_rate.set_xlim(
                        daily_win["trade_date"].min(),
                        daily_win["trade_date"].max(),
                    )
                    ax_pnl_rate.set_ylim(35, 65)
                    ax_pnl_rate.set_xlabel("Day")
                    ax_pnl_rate.set_ylabel("Win rate (%)")
                    ax_pnl_rate.axhline(y=50, color="gray", linewidth=0.8, alpha=0.4)
                    if walk_forward_dt is not None:
                        ax_pnl_rate.axvline(x=float(date2num(walk_forward_dt)), color="orange", linestyle="--", linewidth=1.2)
                    avg_win_rate = float(daily_win["win_rate_30d"].mean())
                    ax_pnl_rate.set_title(f"{rolling_window}-day rolling win rate (%) | Avg: {avg_win_rate:.1f}%")
                else:
                    ax_pnl_rate.text(0.5, 0.5, "No trades after learner_end", ha="center", va="center")
                    ax_pnl_rate.set_axis_off()
            else:
                ax_pnl_rate.text(0.5, 0.5, "No trades available", ha="center", va="center")
                ax_pnl_rate.set_axis_off()
            ax_pnl_rate.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
            ax_pnl_rate.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
            ax_pnl_rate.tick_params(axis="x", rotation=45)

            # PnL distribution histogram (only rendered with --extra-plots).
            if extra_plots:
                ax_pnl_hist = fig.add_subplot(gs[5, 0])
                if not trades_df.empty:
                    pnl_values = trades_df["pnl"].dropna()
                    lower, upper = np.quantile(pnl_values, [0.005, 0.995]) if len(pnl_values) >= 2 else (pnl_values.min(), pnl_values.max())
                    n_bins = 61
                    n_neg = n_bins // 2   
                    n_pos = n_bins - n_neg  
                    neg_edges = np.linspace(lower, 0, n_neg + 1)      
                    pos_edges = np.linspace(0, upper, n_pos + 1)      
                    all_edges = sorted(set(neg_edges.tolist() + pos_edges.tolist()))
                    ax_pnl_hist.hist(pnl_values, bins=all_edges, color="#17becf", edgecolor="black")
                    ax_pnl_hist.set_xlim(lower, upper)
                    ax_pnl_hist.set_title("Profit and loss distribution")
                    ax_pnl_hist.set_xlabel("Trade PnL")
                    ax_pnl_hist.set_ylabel("Frequency")
                else:
                    ax_pnl_hist.text(0.5, 0.5, "No trade profit/loss data", ha="center", va="center")
                    ax_pnl_hist.set_axis_off()

            # Score vs win-rate scatter grouped by score bins (only rendered with --extra-plots).
            if extra_plots:
                ax_score_wr = fig.add_subplot(gs[5, 1])
                if not trades_df.empty and "score" in trades_df.columns:
                    valid = trades_df[["score", "pnl"]].dropna()
                    valid["win"] = (valid["pnl"] > 0).astype(int)
                    n_groups = 100
                    if n_groups >= 2 and len(valid) > 0:
                        valid["score_bin"] = pd.qcut(valid["score"], q=n_groups, labels=False, duplicates="drop")
                        grouped = valid.groupby("score_bin", observed=True).agg(
                            mean_score=("score", "mean"),
                            win_rate=("win", "mean"),
                            n_trades=("pnl", "count"),
                        )
                        ax_score_wr.scatter(grouped["mean_score"], grouped["win_rate"] * 100, s=grouped["n_trades"].clip(upper=50) / 2 + 4, alpha=0.7, color="#d62728")
                        score_pad = (grouped["mean_score"].max() - grouped["mean_score"].min()) * 0.05 or 1.0
                        wr_min = grouped["win_rate"] * 100
                        wr_pad = (wr_min.max() - wr_min.min()) * 0.15 or 2.0
                        ax_score_wr.set_xlim(grouped["mean_score"].min() - score_pad, grouped["mean_score"].max() + score_pad)
                        ax_score_wr.set_ylim(max(0, wr_min.min() - wr_pad), min(100, wr_min.max() + wr_pad))
                        ax_score_wr.axhline(y=50, color="gray", linewidth=0.8, alpha=0.4)
                        ax_score_wr.set_title("Score vs Win Rate (grouped)")
                        ax_score_wr.set_xlabel("Model Score")
                        ax_score_wr.set_ylabel("Win Rate %")
                    else:
                        ax_score_wr.text(0.5, 0.5, "Not enough data", ha="center", va="center")
                        ax_score_wr.set_axis_off()
                else:
                    ax_score_wr.text(0.5, 0.5, "No score data available", ha="center", va="center")
                    ax_score_wr.set_axis_off()

            # 60-day rolling correlation of strategy vs SPY (market dependence).
            rolling_corr = corr_df.dropna(subset=["rolling_corr"])
            if not rolling_corr.empty:
                ax_corr.plot(rolling_corr["trade_date"], rolling_corr["rolling_corr"], color="#9467bd", linewidth=1.5)
                ax_corr.fill_between(rolling_corr["trade_date"], rolling_corr["rolling_corr"], alpha=0.25, color="#9467bd")
                ax_corr.axhline(y=0, color="gray", linewidth=0.8, alpha=0.4)
                ax_corr.set_title("30-day rolling correlation to SPY")
                ax_corr.set_xlabel("Day")
                ax_corr.set_ylabel("Correlation")
                ax_corr.set_ylim(-1, 1)
                ax_corr.xaxis.set_major_locator(MonthLocator(bymonth=[1, 4, 7, 10]))
                ax_corr.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
                ax_corr.tick_params(axis="x", rotation=45)
            else:
                ax_corr.text(0.5, 0.5, "No correlation data available", ha="center", va="center")
                ax_corr.set_axis_off()

            # CAPM scatter of strategy vs SPY daily returns with the regression line.
            if len(capm_df) > 2:
                ax_capm.scatter(x, y, alpha=0.5, s=12, color="#1f77b4", label="Daily returns")
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax_capm.plot(x_fit, capm_alpha_daily + capm_beta * x_fit, color="#d62728",
                             linewidth=2, label="CAPM regression line")
                ax_capm.axhline(0, color="gray", linewidth=0.8, alpha=0.4)
                ax_capm.axvline(0, color="gray", linewidth=0.8, alpha=0.4)
                ax_capm.set_xlabel("SPY daily return")
                ax_capm.set_ylabel("Strategy daily return")
                ax_capm.set_title("CAPM regression: Strategy vs SPY")
                ax_capm.text(
                    0.02, 0.98,
                    f"α (ann.) = {capm_alpha_annual:.2%}\nβ = {capm_beta:.3f}\nR² = {capm_r2:.3f}",
                    transform=ax_capm.transAxes, va="top", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )
                ax_capm.legend(loc="lower right")
            else:
                ax_capm.text(0.5, 0.5, "No CAPM data available", ha="center", va="center")
                ax_capm.set_axis_off()

            fig.tight_layout(pad=2.0)
            fig.subplots_adjust(hspace=0.45, wspace=0.18)
            plt.savefig("results/strategy_assessment.png", bbox_inches="tight")
            plt.close()
    else:
        if not slim:
            print("Warning: no SPY benchmark values available for the selected date range; plot not generated.")
    if not plot_df.empty:
        spy_yoy_gain = (float(final_row["spy_value"]) / start_capital) ** (1.0 / years) - 1.0
        spy_returns = value_df["spy_value"].pct_change().dropna()
        spy_return_std = spy_returns.std(ddof=1)
        spy_sharpe = (
            spy_returns.mean() / spy_return_std * np.sqrt(252)
            if spy_return_std and not pd.isna(spy_return_std)
            else float("nan")
        )
        downside_spy = spy_returns[spy_returns < 0]
        spy_downside_std = downside_spy.std(ddof=1) if len(downside_spy) > 1 else float("nan")
        spy_sortino = (
            spy_returns.mean() / spy_downside_std * np.sqrt(252)
            if not pd.isna(spy_downside_std) and spy_downside_std != 0
            else float("nan")
        )

        if not slim:
            print(f"Final strategy portfolio value: ${final_row['end_value']:,.2f}")
            print(f"Final SPY benchmark value: ${final_row['spy_value']:,.2f}")
            print(f"Annualized strategy YoY gain: {strategy_yoy_gain:.2%}")
            print(f"Annualized SPY benchmark YoY gain: {spy_yoy_gain:.2%}")
            print(f"Strategy Sharpe ratio: {strategy_sharpe:.2f}")
            print(f"SPY benchmark Sharpe ratio: {spy_sharpe:.2f}")
            if not pd.isna(strategy_sortino):
                print(f"Strategy Sortino ratio: {strategy_sortino:.2f}")
            else:
                print("Strategy Sortino ratio: N/A (insufficient downside data)")
            if not pd.isna(spy_sortino):
                print(f"SPY benchmark Sortino ratio: {spy_sortino:.2f}")
            else:
                print("SPY benchmark Sortino ratio: N/A (insufficient downside data)")
            if not pd.isna(capm_alpha_daily):
                print(f"CAPM Alpha (annualized): {capm_alpha_annual:.2%} (daily: {capm_alpha_daily:.4%})")
                print(f"CAPM Beta vs SPY: {capm_beta:.3f}")
                print(f"CAPM R-squared: {capm_r2:.3f}")
                print(f"Covariance (strategy, SPY daily returns): {capm_cov:.6f}")
            else:
                print("CAPM regression: N/A (insufficient aligned return data)")
    return daily_df, trades_df


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper trader backtest using a lightweight linear policy")
    p.add_argument("--mentions-csv", default="ticker_mention_outputs/ticker_counts_all_days_matrix.csv")
    p.add_argument("--db", default="sp500_prices.db")
    p.add_argument("--learner-start", default="2022-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--learner-end", default="2023-01-01")
    p.add_argument("--top-k", type=int, default=0, help="Max number of tickers to trade each day during trading time (0=no limit)")
    p.add_argument("--position-cap", type=float, default=1.0, help="Max fraction of portfolio to allocate to any single ticker (default 1.0 = no cap)")
    p.add_argument("--z-threshold", type=float, default=None, help="Optional minimum mention z-score to qualify for trades")
    p.add_argument("--min-mentions", type=int, default=10, help="Optional minimum mentions to qualify for trades. Default = 5 to avoid low-mention noise.")
    p.add_argument("--start-capital", type=float, default=100000.0, help="Starting capital for the backtest")
    p.add_argument("--transaction-cost", type=float, default=0.000028, help="Fractional transaction cost per trade. Default is FINRA and SEC fees - assume zero commision brokerage account")
    overnight_group = p.add_mutually_exclusive_group()
    overnight_group.add_argument("--buy-spy-overnight", action="store_true", help="Invest the portfolio in SPY from each trading day's close to the next trading day's open")
    overnight_group.add_argument("--hold-overnight", action="store_true", help="Hold selected stocks overnight and sell them at the next trading day's open")
    p.add_argument("--update-policy-fit-daily", action="store_true", help="Re-fit the linear policy each trading day using the accumulated daily training rows")
    p.add_argument("--refresh-data", action="store_true", default=False, help="Run the preflight data refresh scripts before the backtest")
    p.add_argument("--dark-mode", action="store_true", help="Render plots using a dark background style")
    p.add_argument("--slim", action="store_true", help="Suppress console output and skip plot generation while still writing all result CSV files")
    p.add_argument("--extra-plots", action="store_true", help="Render the extra PnL distribution and Score vs Win Rate panels (default: excluded)")
    p.add_argument("--disable-top-mentions", action="store_true", help="Skip trading the ticker that had the most mentions on each day")
    p.add_argument("--allow-shorting", action="store_true", help="Allow short positions for tickers with negative model scores")
    p.add_argument("--position-cap-short", type=float, default=1.0, help="Max fraction of portfolio to allocate per short ticker (only when --allow-shorting is set; default 0.1)")
    p.add_argument("--rolling-window", type=int, default=90, help="N-day rolling win rate window and start offset after learner_end")
    p.add_argument("--walk-forward", nargs="?", const="2026-01-01", default=None, metavar="DATE", help="Perform a walk-forward test starting at DATE (default 2026-01-01). An orange dashed line marks the split in the strategy assessment plot.")
    return p


def run_preflight_scripts(slim: bool = False) -> None:
    commands = [
        # Signal generating scripts here
        [sys.executable, "sp500_db_cli.py", "update", "--db", "sp500_prices.db", "--cache", "sp500_constituents.json", "--refresh-constituents"],
    ]
    for command in commands:
        if not slim:
            print(f"Running: {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Preflight command failed with exit code {completed.returncode}: {' '.join(command)}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.refresh_data:
        run_preflight_scripts(slim=args.slim)
    mentions = load_mentions_matrix(args.mentions_csv)
    run_backtest(
        mentions_df=mentions,
        db_path=args.db,
        learner_start=args.learner_start,
        end=args.end,
        learner_end=args.learner_end,
        top_k=args.top_k,
        z_threshold=args.z_threshold,
        min_mentions=args.min_mentions,
        position_cap=args.position_cap,
        start_capital=args.start_capital,
        transaction_cost=args.transaction_cost,
        buy_spy_overnight=args.buy_spy_overnight,
        hold_overnight=args.hold_overnight,
        update_policy_fit_daily=args.update_policy_fit_daily,
        allow_shorting=args.allow_shorting,
        position_cap_short=args.position_cap_short,
        rolling_window=args.rolling_window,
        walk_forward_date=args.walk_forward,
        slim=args.slim,
        extra_plots=args.extra_plots,
        disable_top_mentions=args.disable_top_mentions,
    )
    if not args.slim:
        print("Linear policy backtest complete. Results saved in results/*.csv and results/strategy_assessment.png")


if __name__ == "__main__":
    main()