#!/usr/bin/env python3
"""
Run Examples:
  python sp500_db_cli.py init --db sp500_prices.db --cache sp500_constituents.json --start 2022-01-01
  python sp500_db_cli.py update --db sp500_prices.db --cache sp500_constituents.json --refresh-constituents
  python sp500_db_cli.py query --db sp500_prices.db --ticker GOOG --start 2025-01-01 --end 2026-01-01
  python sp500_db_cli.py chart --db sp500_prices.db --ticker SPY --start 2025-01-01 --out charts/spy.png
  python sp500_db_cli.py day-change --db sp500_prices.db --ticker AAPL --date 2025-01-02
  python sp500_db_cli.py show-groups --db sp500_prices.db
"""

import argparse
import concurrent.futures
import json
import os
import random
import sqlite3
import time
from datetime import datetime, timedelta, date
from io import StringIO
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf

try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DATAHUB_SP500_CSV = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"

DEFAULT_DB = "sp500_prices.db"
DEFAULT_CACHE = "sp500_constituents.json"
DEFAULT_START = "2022-01-01"

# Number of concurrent Yahoo fetches during `update`. The workload is network-bound, so threading gives a near-linear speedup.
PARALLEL_FETCH_WORKERS = 4

SP500_TRACKER_TICKER = "SPY"  
SP500_TRACKER_NAME = "SPDR S&P 500 ETF Trust"

SHARE_CLASS_GROUPS = {
    "GOOGL": ["GOOG"],   # Alphabet
    "FOX": ["FOXA"],     # Fox
    "NWS": ["NWSA"],     # News Corp
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    )
}


# ---------------------------
# Utilities
# ---------------------------

def log(msg: str) -> None:
    print(msg, flush=True)

def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def normalize_ticker_from_source(t: str) -> str:
    return t.replace(".", "-").strip().upper()

def normalize_ticker_for_yf(t: str) -> str:
    return t.replace(".", "-").strip().upper()

def request_with_retry(url: str, timeout: int = 20, retries: int = 4, backoff_base: float = 1.4) -> requests.Response:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = (backoff_base ** attempt) + random.uniform(0, 0.8)
                log(f"[NET] attempt {attempt}/{retries} failed for {url}: {e} | retrying in {sleep_s:.1f}s")
                time.sleep(sleep_s)
            else:
                log(f"[NET] attempt {attempt}/{retries} failed for {url}: {e}")
    raise RuntimeError(f"Request failed after retries: {url}; last_error={last_err}")


# ---------------------------
# Constituents fetch/cache
# ---------------------------

def fetch_sp500_from_wikipedia() -> Dict[str, str]:
    resp = request_with_retry(WIKI_SP500_URL)
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("Wikipedia parse produced no tables")
    df = tables[0].copy()
    if "Symbol" not in df.columns or "Security" not in df.columns:
        raise RuntimeError("Wikipedia schema changed: expected Symbol/Security")
    df["Symbol"] = df["Symbol"].map(normalize_ticker_from_source)
    d = dict(zip(df["Symbol"], df["Security"]))
    if not d:
        raise RuntimeError("Wikipedia produced empty ticker dictionary")
    return d

def fetch_sp500_from_datahub() -> Dict[str, str]:
    resp = request_with_retry(DATAHUB_SP500_CSV)
    df = pd.read_csv(StringIO(resp.text))
    if "Symbol" not in df.columns or "Name" not in df.columns:
        raise RuntimeError("DataHub schema changed: expected Symbol/Name")
    df["Symbol"] = df["Symbol"].map(normalize_ticker_from_source)
    d = dict(zip(df["Symbol"], df["Name"]))
    if not d:
        raise RuntimeError("DataHub produced empty ticker dictionary")
    return d

def save_constituents_cache(cache_path: str, ticker_dict: Dict[str, str], source: str) -> None:
    payload = {
        "saved_at_local": datetime.now().isoformat(),
        "source": source,
        "count": len(ticker_dict),
        "tickers": ticker_dict
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def load_constituents_cache(cache_path: str) -> Optional[Dict[str, str]]:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tickers = payload.get("tickers", {})
        if isinstance(tickers, dict) and tickers:
            return {normalize_ticker_from_source(k): str(v) for k, v in tickers.items()}
    except FileNotFoundError:
        return None
    except Exception as e:
        log(f"[WARN] Failed to read cache {cache_path}: {e}")
    return None

def get_constituents_robust(cache_path: str, force_refresh: bool = False) -> Tuple[Dict[str, str], str]:
    """
    Returns: (ticker_dict, source_used)
    source_used in {"wikipedia", "datahub", "cache"}
    """
    if not force_refresh:
        try:
            d = fetch_sp500_from_wikipedia()
            save_constituents_cache(cache_path, d, "wikipedia")
            return d, "wikipedia"
        except Exception as e:
            log(f"[WARN] Wikipedia failed: {e}")

        try:
            d = fetch_sp500_from_datahub()
            save_constituents_cache(cache_path, d, "datahub")
            return d, "datahub"
        except Exception as e:
            log(f"[WARN] DataHub failed: {e}")

        cached = load_constituents_cache(cache_path)
        if cached:
            log(f"[WARN] Using cached constituents from {cache_path}")
            return cached, "cache"

        raise RuntimeError("All constituents sources failed and no cache available.")

    errors = []
    try:
        d = fetch_sp500_from_wikipedia()
        save_constituents_cache(cache_path, d, "wikipedia")
        return d, "wikipedia"
    except Exception as e:
        errors.append(f"Wikipedia: {e}")

    try:
        d = fetch_sp500_from_datahub()
        save_constituents_cache(cache_path, d, "datahub")
        return d, "datahub"
    except Exception as e:
        errors.append(f"DataHub: {e}")

    cached = load_constituents_cache(cache_path)
    if cached:
        log("[WARN] force_refresh failed on web sources; using cache as fallback.")
        return cached, "cache"

    raise RuntimeError("force_refresh requested, all web sources failed, and no cache exists: " + " | ".join(errors))


# ---------------------------
# DB
# ---------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sp500_companies (
        ticker TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        canonical_ticker TEXT NOT NULL,
        is_alias INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS alias_map (
        alias_ticker TEXT PRIMARY KEY,
        canonical_ticker TEXT NOT NULL,
        FOREIGN KEY (canonical_ticker) REFERENCES sp500_companies(ticker) ON DELETE CASCADE
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS daily_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,              -- canonical ticker only
        trade_date TEXT NOT NULL,          -- YYYY-MM-DD
        open_price REAL NOT NULL,
        close_price REAL NOT NULL,
        source TEXT NOT NULL DEFAULT 'yfinance',
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        FOREIGN KEY (ticker) REFERENCES sp500_companies(ticker) ON DELETE CASCADE,
        UNIQUE (ticker, trade_date)
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS sp500_tracker_prices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tracker_ticker TEXT NOT NULL,      -- e.g., SPY or ^GSPC
        trade_date TEXT NOT NULL,          -- YYYY-MM-DD
        open_price REAL NOT NULL,
        close_price REAL NOT NULL,
        source TEXT NOT NULL DEFAULT 'yfinance',
        updated_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE (tracker_ticker, trade_date)
    );
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    """)
    conn.commit()

def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("""
    INSERT INTO meta (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value=excluded.value;
    """, (key, value))
    conn.commit()

def build_canonical_and_alias_maps(ticker_dict: Dict[str, str]):
    all_tickers = set(ticker_dict.keys())
    alias_to_canonical = {}

    for canonical, aliases in SHARE_CLASS_GROUPS.items():
        c = normalize_ticker_from_source(canonical)
        if c in all_tickers:
            for a in aliases:
                aa = normalize_ticker_from_source(a)
                if aa in all_tickers:
                    alias_to_canonical[aa] = c

    companies_rows = []
    alias_rows = []

    for ticker, name in ticker_dict.items():
        canonical = alias_to_canonical.get(ticker, ticker)
        is_alias = 1 if ticker in alias_to_canonical else 0
        companies_rows.append((ticker, name, canonical, is_alias))
        if is_alias:
            alias_rows.append((ticker, canonical))

    canonical_universe = sorted({row[2] for row in companies_rows})
    return companies_rows, alias_rows, canonical_universe

def persist_companies(conn: sqlite3.Connection, companies_rows, alias_rows) -> None:
    conn.executemany("""
    INSERT INTO sp500_companies (ticker, company_name, canonical_ticker, is_alias)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(ticker) DO UPDATE SET
      company_name=excluded.company_name,
      canonical_ticker=excluded.canonical_ticker,
      is_alias=excluded.is_alias;
    """, companies_rows)

    conn.executemany("""
    INSERT INTO alias_map (alias_ticker, canonical_ticker)
    VALUES (?, ?)
    ON CONFLICT(alias_ticker) DO UPDATE SET
      canonical_ticker=excluded.canonical_ticker;
    """, alias_rows)

    conn.commit()

def get_canonical_universe(conn: sqlite3.Connection) -> List[str]:
    cur = conn.execute("""
    SELECT DISTINCT canonical_ticker
    FROM sp500_companies
    ORDER BY canonical_ticker;
    """)
    return [r[0] for r in cur.fetchall()]

def resolve_to_canonical(conn: sqlite3.Connection, ticker: str) -> Optional[str]:
    t = normalize_ticker_from_source(ticker)
    cur = conn.execute("SELECT canonical_ticker FROM sp500_companies WHERE ticker=?;", (t,))
    r = cur.fetchone()
    return r[0] if r else None


# ---------------------------
# Prices
# ---------------------------

def fetch_open_close(ticker: str, start_date: str, end_date: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                normalize_ticker_for_yf(ticker),
                start=start_date,
                end=end_date,          # end-exclusive for yfinance
                auto_adjust=False,
                progress=False,
                threads=False,
                interval="1d"
            )
            if df.empty:
                return pd.DataFrame(columns=["trade_date", "open_price", "close_price"])
            out = df[["Open", "Close"]].copy()
            out.columns = ["open_price", "close_price"]
            out.index = pd.to_datetime(out.index).date
            out = out.reset_index().rename(columns={"index": "trade_date"})
            out["trade_date"] = out["trade_date"].astype(str)
            return out
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = (1.5 ** attempt) + random.uniform(0, 0.7)
                time.sleep(sleep_s)
            else:
                raise RuntimeError(f"yfinance failed for {ticker}: {last_err}")

def upsert_prices(
    conn: sqlite3.Connection,
    canonical_ticker: str,
    prices_df: pd.DataFrame,
    commit: bool = True
) -> int:
    if prices_df.empty:
        return 0
    rows = [
        (canonical_ticker, r.trade_date, float(r.open_price), float(r.close_price), "yfinance")
        for r in prices_df.itertuples(index=False)
        if pd.notnull(r.open_price) and pd.notnull(r.close_price)
    ]
    conn.executemany("""
    INSERT INTO daily_prices (ticker, trade_date, open_price, close_price, source)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(ticker, trade_date) DO UPDATE SET
      open_price=excluded.open_price,
      close_price=excluded.close_price,
      source=excluded.source,
      updated_at=datetime('now');
    """, rows)
    if commit:
        conn.commit()
    return len(rows)

def get_latest_trade_date(conn: sqlite3.Connection, canonical_ticker: str) -> Optional[date]:
    cur = conn.execute("SELECT MAX(trade_date) FROM daily_prices WHERE ticker=?;", (canonical_ticker,))
    row = cur.fetchone()
    if row and row[0]:
        return datetime.strptime(row[0], "%Y-%m-%d").date()
    return None

def upsert_tracker_prices(conn: sqlite3.Connection, tracker_ticker: str, prices_df: pd.DataFrame) -> int:
    if prices_df.empty:
        return 0
    rows = [
        (tracker_ticker, r.trade_date, float(r.open_price), float(r.close_price), "yfinance")
        for r in prices_df.itertuples(index=False)
        if pd.notnull(r.open_price) and pd.notnull(r.close_price)
    ]
    conn.executemany("""
    INSERT INTO sp500_tracker_prices (tracker_ticker, trade_date, open_price, close_price, source)
    VALUES (?, ?, ?, ?, ?)
    ON CONFLICT(tracker_ticker, trade_date) DO UPDATE SET
      open_price=excluded.open_price,
      close_price=excluded.close_price,
      source=excluded.source,
      updated_at=datetime('now');
    """, rows)
    conn.commit()
    return len(rows)

def get_latest_tracker_trade_date(conn: sqlite3.Connection, tracker_ticker: str) -> Optional[date]:
    cur = conn.execute("""
    SELECT MAX(trade_date) FROM sp500_tracker_prices WHERE tracker_ticker=?;
    """, (tracker_ticker,))
    row = cur.fetchone()
    if row and row[0]:
        return datetime.strptime(row[0], "%Y-%m-%d").date()
    return None

def run_day_change(conn: sqlite3.Connection, ticker: str, day: str) -> None:
    """
    Compute intraday % change for a ticker on a given trading day:
    ((close - open) / open) * 100

    Works for:
    - tracker tickers (e.g. SPY) via sp500_tracker_prices
    - company tickers (including aliases) via daily_prices
    """
    t = normalize_ticker_from_source(ticker)

    # 1) Try tracker table first
    cur = conn.execute("""
    SELECT tracker_ticker, trade_date, open_price, close_price
    FROM sp500_tracker_prices
    WHERE tracker_ticker=? AND trade_date=?
    LIMIT 1;
    """, (t, day))
    row = cur.fetchone()

    source = "sp500_tracker_prices"
    resolved = t

    # 2) Fallback to company (alias -> canonical)
    if row is None:
        canonical = resolve_to_canonical(conn, t)
        if not canonical:
            log(f"Ticker {ticker} not found in tracker table or sp500_companies.")
            return

        cur = conn.execute("""
        SELECT ticker, trade_date, open_price, close_price
        FROM daily_prices
        WHERE ticker=? AND trade_date=?
        LIMIT 1;
        """, (canonical, day))
        row = cur.fetchone()
        source = "daily_prices"
        resolved = canonical

    if row is None:
        log(f"No data found for ticker={t} on date={day}. (Maybe non-trading day?)")
        return

    _, trade_date, open_price, close_price = row
    open_price = float(open_price)
    close_price = float(close_price)

    if open_price == 0:
        log(f"Cannot compute percentage change because open price is 0 for {resolved} on {trade_date}.")
        return

    pct_change = ((close_price - open_price) / open_price) * 100.0

    log(f"input={t} resolved={resolved} source={source} date={trade_date}")
    log(f"open={open_price:.4f} close={close_price:.4f} day_change_pct={pct_change:.4f}%")


# ---------------------------
# Command implementations
# ---------------------------

def refresh_constituents(conn: sqlite3.Connection, cache_path: str, force_refresh: bool) -> None:
    ticker_dict, source = get_constituents_robust(cache_path=cache_path, force_refresh=force_refresh)
    companies_rows, alias_rows, canonical_universe = build_canonical_and_alias_maps(ticker_dict)
    persist_companies(conn, companies_rows, alias_rows)

    set_meta(conn, "sp500_snapshot_time_local", datetime.now().isoformat())
    set_meta(conn, "sp500_source", source)
    set_meta(conn, "sp500_count_total_tickers", str(len(ticker_dict)))
    set_meta(conn, "sp500_count_canonical_tickers", str(len(canonical_universe)))
    set_meta(conn, "sp500_cache_path", cache_path)
    set_meta(conn, "sp500_tracker_ticker", SP500_TRACKER_TICKER)
    set_meta(conn, "sp500_tracker_name", SP500_TRACKER_NAME)

    log(f"[UNIVERSE] source={source} total={len(ticker_dict)} canonical={len(canonical_universe)} aliases={len(alias_rows)}")

def initialize_history(conn: sqlite3.Connection, start: str, end_inclusive: str) -> None:
    # yfinance end is exclusive, so add 1 day
    end_exclusive = (datetime.strptime(end_inclusive, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()

    tickers = get_canonical_universe(conn)
    total = 0
    for i, t in enumerate(tickers, 1):
        try:
            df = fetch_open_close(t, start, end_exclusive)
            n = upsert_prices(conn, t, df)
            total += n
            if i % 25 == 0 or i == len(tickers):
                log(f"[INIT] {i}/{len(tickers)} processed | rows upserted={total}")
        except Exception as e:
            log(f"[INIT][ERROR] {t}: {e}")

    # Backfill S&P 500 tracker
    try:
        tracker_df = fetch_open_close(SP500_TRACKER_TICKER, start, end_exclusive)
        tracker_rows = upsert_tracker_prices(conn, SP500_TRACKER_TICKER, tracker_df)
        log(f"[INIT][TRACKER] {SP500_TRACKER_TICKER} rows upserted={tracker_rows}")
    except Exception as e:
        log(f"[INIT][TRACKER][ERROR] {SP500_TRACKER_TICKER}: {e}")

    set_meta(conn, "last_full_init_at", datetime.now().isoformat())
    set_meta(conn, "last_daily_update_at", datetime.now().isoformat())
    log(f"[INIT] done | total rows upserted={total}")

def run_daily_update(conn: sqlite3.Connection, from_date_if_empty: str = DEFAULT_START) -> None:
    tickers = get_canonical_universe(conn)
    end_exclusive = datetime.now().date() + timedelta(days=1)
    default_start = datetime.strptime(from_date_if_empty, "%Y-%m-%d").date()

    # One batched query for the last stored trade date of every ticker
    # (replaces one lookup per ticker, and powers the "only fetch missing" logic).
    cur = conn.execute("SELECT ticker, MAX(trade_date) FROM daily_prices GROUP BY ticker")
    latest_by_ticker = {r[0]: r[1] for r in cur.fetchall()}

    # Build the (ticker, start) windows that may contain missing trading days.
    jobs = []
    for t in tickers:
        latest_s = latest_by_ticker.get(t)
        latest = datetime.strptime(latest_s, "%Y-%m-%d").date() if latest_s else None
        start = default_start if latest is None else latest + timedelta(days=1)
        if start < end_exclusive:
            jobs.append((t, start))

    if not jobs:
        log("[UPDATE] all tickers already up to date; nothing to fetch")
        set_meta(conn, "last_daily_update_at", datetime.now().isoformat())
        log("[UPDATE] done | total rows upserted=0")
        return

    # Only request windows that actually contain NYSE trading days, so weekend /
    # holiday runs (or DBs current through the last trading day) make no requests.
    trading_days = None
    if mcal is not None:
        try:
            min_start = min(start for _, start in jobs)
            calendar = mcal.get_calendar("NYSE")
            valid = calendar.valid_days(
                start_date=min_start.isoformat(),
                end_date=(end_exclusive - timedelta(days=1)).isoformat()
            )
            trading_days = {d.date() for d in valid}
        except Exception as e:
            log(f"[UPDATE][WARN] market-calendar lookup failed ({e}); fetching all windows")

    fetch_jobs = []
    for t, start in jobs:
        if trading_days is not None:
            if not any(start <= d < end_exclusive for d in trading_days):
                continue  # window contains no trading days -> nothing to fetch
        fetch_jobs.append((t, start))

    if not fetch_jobs:
        log("[UPDATE] no missing trading days; nothing to fetch")
        set_meta(conn, "last_daily_update_at", datetime.now().isoformat())
        log("[UPDATE] done | total rows upserted=0")
        return

    # Parallel fetch phase. Workers only hit the network; the DB connection is
    # never touched from worker threads (SQLite uses a single connection here).
    results = []  # (ticker, prices_df)
    done = 0
    total_jobs = len(fetch_jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARALLEL_FETCH_WORKERS) as executor:
        futures = {
            executor.submit(fetch_open_close, t, start.isoformat(), end_exclusive.isoformat()): t
            for t, start in fetch_jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            done += 1
            try:
                df = fut.result()
                results.append((t, df))
            except Exception as e:
                log(f"[UPDATE][ERROR] {t}: {e}")
            if done % 25 == 0 or done == total_jobs:
                log(f"[UPDATE] {done}/{total_jobs} fetched")

    # Sequential upsert phase on the main thread; one commit at the end.
    total = 0
    for t, df in results:
        total += upsert_prices(conn, t, df, commit=False)
    conn.commit()

    # Update tracker incrementally
    try:
        latest_tracker = get_latest_tracker_trade_date(conn, SP500_TRACKER_TICKER)
        start_tracker = default_start if latest_tracker is None else latest_tracker + timedelta(days=1)

        if start_tracker < end_exclusive:
            tracker_df = fetch_open_close(
                SP500_TRACKER_TICKER,
                start_tracker.isoformat(),
                end_exclusive.isoformat()
            )
            tracker_rows = upsert_tracker_prices(conn, SP500_TRACKER_TICKER, tracker_df)
            log(f"[UPDATE][TRACKER] {SP500_TRACKER_TICKER} rows upserted={tracker_rows}")
    except Exception as e:
        log(f"[UPDATE][TRACKER][ERROR] {SP500_TRACKER_TICKER}: {e}")

    set_meta(conn, "last_daily_update_at", datetime.now().isoformat())
    log(f"[UPDATE] done | total rows upserted={total}")

def run_query_unified(conn: sqlite3.Connection, ticker: str, start: str, end: str, limit: int) -> None:
    t = normalize_ticker_from_source(ticker)

    # tracker first
    cur = conn.execute("""
    SELECT tracker_ticker AS resolved_ticker, trade_date, open_price, close_price
    FROM sp500_tracker_prices
    WHERE tracker_ticker=? AND trade_date BETWEEN ? AND ?
    ORDER BY trade_date
    LIMIT ?;
    """, (t, start, end, limit))
    rows = cur.fetchall()

    if rows:
        log(f"input={t} resolved={t} source=sp500_tracker_prices rows={len(rows)}")
        for r in rows:
            log(str(r))
        return

    # company alias->canonical
    canonical = resolve_to_canonical(conn, t)
    if not canonical:
        log(f"Ticker {ticker} not found in tracker table or sp500_companies.")
        return

    cur = conn.execute("""
    SELECT ticker AS resolved_ticker, trade_date, open_price, close_price
    FROM daily_prices
    WHERE ticker=? AND trade_date BETWEEN ? AND ?
    ORDER BY trade_date
    LIMIT ?;
    """, (canonical, start, end, limit))
    rows = cur.fetchall()

    log(f"input={t} resolved={canonical} source=daily_prices rows={len(rows)}")
    for r in rows:
        log(str(r))

def run_chart_unified(
    conn: sqlite3.Connection,
    ticker: str,
    start: str,
    end: str,
    out: str,
    use_close: bool = True
) -> None:
    t = normalize_ticker_from_source(ticker)
    label = "Close" if use_close else "Open"

    # tracker first
    cur = conn.execute("""
    SELECT trade_date, open_price, close_price
    FROM sp500_tracker_prices
    WHERE tracker_ticker=? AND trade_date BETWEEN ? AND ?
    ORDER BY trade_date;
    """, (t, start, end))
    rows = cur.fetchall()

    series_name = t
    source_table = "sp500_tracker_prices"

    # company fallback
    if not rows:
        canonical = resolve_to_canonical(conn, t)
        if not canonical:
            log(f"Ticker {ticker} not found in tracker table or sp500_companies.")
            return

        cur = conn.execute("""
        SELECT trade_date, open_price, close_price
        FROM daily_prices
        WHERE ticker=? AND trade_date BETWEEN ? AND ?
        ORDER BY trade_date;
        """, (canonical, start, end))
        rows = cur.fetchall()
        series_name = canonical
        source_table = "daily_prices"

    if not rows:
        log(f"No data found for {ticker} between {start} and {end}.")
        return

    dates = [datetime.strptime(r[0], "%Y-%m-%d").date() for r in rows]
    open_prices = [float(r[1]) for r in rows]
    close_prices = [float(r[2]) for r in rows]
    y = close_prices if use_close else open_prices

    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(dates, y, linewidth=1.8, label=f"{series_name} {label}")
    plt.title(f"{series_name} {label} Price ({start} to {end})")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    plt.close()

    log(f"[CHART] Saved chart to: {out}")
    log(f"[CHART] input={t} resolved={series_name} source={source_table} points={len(rows)}")

def run_show_groups(conn: sqlite3.Connection) -> None:
    cur = conn.execute("""
    SELECT canonical_ticker, GROUP_CONCAT(ticker, ', ') AS members
    FROM sp500_companies
    GROUP BY canonical_ticker
    HAVING COUNT(*) > 1
    ORDER BY canonical_ticker;
    """)
    rows = cur.fetchall()
    if not rows:
        log("No multi-ticker canonical groups found.")
        return
    for canonical, members in rows:
        log(f"{canonical}: {members}")


# ---------------------------
# CLI
# ---------------------------

def cmd_init(args):
    conn = get_connection(args.db)
    init_db(conn)
    log("[INIT] refreshing constituents...")
    refresh_constituents(conn, cache_path=args.cache, force_refresh=True)

    end_inclusive = args.end if args.end else today_str()
    initialize_history(conn, start=args.start, end_inclusive=end_inclusive)
    conn.close()

def cmd_update(args):
    conn = get_connection(args.db)
    init_db(conn)

    if args.refresh_constituents:
        log("[UPDATE] refreshing constituents...")
        refresh_constituents(conn, cache_path=args.cache, force_refresh=True)
    else:
        if not get_canonical_universe(conn):
            log("[UPDATE] no universe in DB, loading constituents (non-forced)...")
            refresh_constituents(conn, cache_path=args.cache, force_refresh=False)

    run_daily_update(conn, from_date_if_empty=args.from_date_if_empty)
    conn.close()

def cmd_refresh_constituents(args):
    conn = get_connection(args.db)
    init_db(conn)
    refresh_constituents(conn, cache_path=args.cache, force_refresh=args.force)
    conn.close()

def cmd_query(args):
    conn = get_connection(args.db)
    init_db(conn)
    end_value = args.end if args.end else today_str()
    run_query_unified(
        conn=conn,
        ticker=args.ticker,
        start=args.start,
        end=end_value,
        limit=args.limit
    )
    conn.close()

def cmd_chart(args):
    conn = get_connection(args.db)
    init_db(conn)
    end_value = args.end if args.end else today_str()
    run_chart_unified(
        conn=conn,
        ticker=args.ticker,
        start=args.start,
        end=end_value,
        out=args.out,
        use_close=(not args.use_open)
    )
    conn.close()

def cmd_day_change(args):
    conn = get_connection(args.db)
    init_db(conn)
    run_day_change(conn, ticker=args.ticker, day=args.date)
    conn.close()

def cmd_show_groups(args):
    conn = get_connection(args.db)
    init_db(conn)
    run_show_groups(conn)
    conn.close()

def build_parser():
    p = argparse.ArgumentParser(description="Robust S&P 500 open/close DB CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Init DB, refresh constituents, backfill history")
    p_init.add_argument("--db", default=DEFAULT_DB)
    p_init.add_argument("--cache", default=DEFAULT_CACHE)
    p_init.add_argument("--start", default=DEFAULT_START)
    p_init.add_argument("--end", required=False, default=None, help="Inclusive end date YYYY-MM-DD (default: today)")
    p_init.set_defaults(func=cmd_init)

    p_upd = sub.add_parser("update", help="Daily incremental update")
    p_upd.add_argument("--db", default=DEFAULT_DB)
    p_upd.add_argument("--cache", default=DEFAULT_CACHE)
    p_upd.add_argument("--refresh-constituents", action="store_true")
    p_upd.add_argument("--from-date-if-empty", default=DEFAULT_START)
    p_upd.set_defaults(func=cmd_update)

    p_ref = sub.add_parser("refresh-constituents", help="Refresh constituents only")
    p_ref.add_argument("--db", default=DEFAULT_DB)
    p_ref.add_argument("--cache", default=DEFAULT_CACHE)
    p_ref.add_argument("--force", action="store_true", help="Force web refresh; fallback to cache if unavailable")
    p_ref.set_defaults(func=cmd_refresh_constituents)

    p_q = sub.add_parser("query", help="Unified query for company/alias or tracker ticker (e.g. SPY)")
    p_q.add_argument("--db", default=DEFAULT_DB)
    p_q.add_argument("--ticker", required=True)
    p_q.add_argument("--start", required=True)
    p_q.add_argument("--end", required=False, default=None, help="YYYY-MM-DD (default: today)")
    p_q.add_argument("--limit", type=int, default=100)
    p_q.set_defaults(func=cmd_query)

    p_c = sub.add_parser("chart", help="Create chart for ticker (company alias/canonical or tracker like SPY)")
    p_c.add_argument("--db", default=DEFAULT_DB)
    p_c.add_argument("--ticker", required=True)
    p_c.add_argument("--start", required=True)
    p_c.add_argument("--end", required=False, default=None, help="YYYY-MM-DD (default: today)")
    p_c.add_argument("--out", default="charts/price_chart.png")
    p_c.add_argument("--use-open", action="store_true", help="Chart open price instead of close price")
    p_c.set_defaults(func=cmd_chart)

    p_dc = sub.add_parser("day-change", help="Compute percent change from open to close for a ticker on a given date")
    p_dc.add_argument("--db", default=DEFAULT_DB)
    p_dc.add_argument("--ticker", required=True, help="Ticker symbol (company, alias, or tracker like SPY)")
    p_dc.add_argument("--date", required=True, help="Trading date YYYY-MM-DD")
    p_dc.set_defaults(func=cmd_day_change)

    p_g = sub.add_parser("show-groups", help="Show canonical/alias groups")
    p_g.add_argument("--db", default=DEFAULT_DB)
    p_g.set_defaults(func=cmd_show_groups)

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()