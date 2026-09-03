"""
alpaca_trader.py — Rebalance an Alpaca trading account toward today's target allocations.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.trading.requests import MarketOrderRequest

import pandas as pd

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "")

TRADES_CSV = os.path.join("results", "trades.csv")

ET = ZoneInfo("America/New_York")
ORDER_TIME_ET = (9, 25)  # (hour, minute) ET — earliest time orders may be submitted
DEFAULT_START_CAPITAL = 10000.0  # note: open new paper accounts with 10k USD
MIN_ALLOC = 1.0  # minimum dollar amount for notional buy/sell orders

GENERATOR_FLAGS: List[str] = [
    "--learner-start", "2022-01-01",
    "--update-policy-fit-daily",
    "--hold-overnight",
    "--slim",
]

REQUIRED_COLUMNS = ["trade_date", "ticker", "side", "alloc", "open"]


def run_generator(script_dir: str, learner_end: str, start_capital: float,
                  refresh: bool = True) -> None:
    # Run trading_model.py to regenerate trades.csv
    script = os.path.join(script_dir, "trading_model.py")
    if not os.path.exists(script):
        sys.exit(f"ERROR: generator script not found: {script}")
    flags = GENERATOR_FLAGS + (["--refresh-data"] if refresh else []) + [
        f"--learner-end={learner_end}",
        f"--start-capital={start_capital:.2f}",
    ]
    print(">> Generating results/trades.csv via trading_model.py ...")
    print(f"   python {os.path.basename(script)} {' '.join(flags)}")
    if refresh:
        print("   (--refresh-data runs the network-bound preflight scripts; this may take a while)")
    cmd = [sys.executable, script] + flags
    try:
        subprocess.run(cmd, cwd=script_dir, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(f"ERROR: trading_model.py exited with code {exc.returncode}")


def previous_trading_day(ref: datetime) -> str:
    # gets most recent NYSE trading day strictly before `ref`, as an ISO date string
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        schedule = nyse.schedule(
            start_date=ref.date() - timedelta(days=14),
            end_date=ref.date(),
        )
        sessions = sorted(s.date() for s in schedule.index if s.date() < ref.date())
        if sessions:
            return str(sessions[-1])
    except Exception as exc:
        print(f"  WARNING: could not resolve NYSE calendar ({exc}); using weekday fallback")
    day = ref.date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return str(day)


def wait_until_order_time() -> None:
    # Block until ORDER_TIME_ET in the ET timezone, then return.
    # If that wall-clock time has already passed today, returns immediately.
    now = datetime.now(ET)
    target = now.replace(
        hour=ORDER_TIME_ET[0], minute=ORDER_TIME_ET[1], second=0, microsecond=0,
    )
    print(f">> Current time (ET): {now:%Y-%m-%d %H:%M:%S}")
    if now >= target:
        print(f"   (already past {ORDER_TIME_ET[0]:02d}:{ORDER_TIME_ET[1]:02d} ET — no wait needed)")
        return
    delay = (target - now).total_seconds()
    print(f">> Waiting {delay / 60:.1f} min until {target:%Y-%m-%d %H:%M:%S} ET ...")
    time.sleep(delay)


def acct_value(account, key: str) -> float:
    # Read a numeric account field, tolerating both the typed account and the raw dict
    if account is None:
        return 0.0
    raw = getattr(account, key, None)
    if raw is None and hasattr(account, "get"):
        raw = account.get(key)
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def print_portfolio(account, positions) -> None:
    # Print the current Alpaca account/position state
    print(">> Current Alpaca portfolio state:")
    if account is None:
        print("   (unavailable)")
        return
    print(f"   Equity          : {acct_value(account, 'equity'):>16,.2f}")
    print(f"   Cash            : {acct_value(account, 'cash'):>16,.2f}")
    print(f"   Buying power    : {acct_value(account, 'buying_power'):>16,.2f}")
    if not positions:
        print("   Positions       : none")
        return
    print(f"   Positions       : {len(positions)}")
    for p in positions:
        side = getattr(p, "side", "long")
        print(f"     {p.symbol:6s} {side:5s} qty={float(p.qty):>10.4f}  "
              f"market_value={float(p.market_value):>12,.2f}  cost_basis={float(p.cost_basis):>12,.2f}")


def load_trades(csv_path: str) -> pd.DataFrame:
    # Load and validate results/trades.csv
    if not os.path.exists(csv_path):
        sys.exit(
            f"ERROR: {csv_path} not found — run trading_model.py first "
            "(or drop --no-generate)."
        )
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: trades.csv is missing required columns: {missing}")
    for col in ("alloc", "open", "close", "exit_price", "return", "pnl", "cost", "mentions", "z_score", "score"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trade_date"] = df["trade_date"].astype(str)
    return df


def select_trades(df: pd.DataFrame, date: Optional[str]) -> pd.DataFrame:
    # Pick the trades to submit: a specific trade_date or the latest one
    if df.empty:
        sys.exit(
            "No trades found in results/trades.csv — no trading days after learner-end (e.g. weekend run). Nothing to submit."
        )
    if date:
        subset = df[df["trade_date"] == date]
        if subset.empty:
            sys.exit(f"ERROR: no trades found for trade_date {date}.")
    else:
        latest = sorted(df["trade_date"].unique())[-1]
        subset = df[df["trade_date"] == latest]
        print(f">> Selected latest trade_date: {latest} ({len(subset)} trade(s))")
    return subset.reset_index(drop=True)


def make_client():
    # Create the Alpaca paper-trading client
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        sys.exit("ERROR: alpaca-py is not installed. Run:  python -m pip install alpaca-py")
    return TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)


def position_value(pos) -> float:
    # Read a position's market_value, tolerating both the typed Position and the raw dict
    if pos is None:
        return 0.0
    if isinstance(pos, dict):
        return float(pos.get("market_value") or 0)
    return float(getattr(pos, "market_value") or 0)


def position_qty(pos) -> float:
    # Read a position's share quantity, tolerating both the typed Position and the raw dict
    if pos is None:
        return 0.0
    if isinstance(pos, dict):
        return float(pos.get("qty") or 0)
    return float(getattr(pos, "qty") or 0)


def submit_market(client, ticker: str, is_buy: bool, dry_run: bool,
                  notional: Optional[float] = None, qty: Optional[float] = None) -> None:
    # Submit a market order — either dollar-sized (notional) or share-sized (qty) 
    tag = "BUY " if is_buy else "SELL"
    side = OrderSide.BUY if is_buy else OrderSide.SELL

    if notional is not None:
        notional = round(float(notional), 2)
        if notional < MIN_ALLOC:
            print(f"  SKIP  {tag}  {ticker:6s} (notional=${notional:,.2f} < min_alloc=${MIN_ALLOC:,.2f})")
            return
        order = MarketOrderRequest(
            symbol=ticker, type=OrderType.MARKET, notional=f"{notional:.2f}",
            side=side, time_in_force=TimeInForce.DAY,
        )
        amount_desc = f"notional=${notional:,.2f}"
    else:
        order = MarketOrderRequest(
            symbol=ticker, type=OrderType.MARKET, qty=qty,
            side=side, time_in_force=TimeInForce.DAY,
        )
        amount_desc = f"qty={qty:g}"

    if dry_run:
        print(f"  [dry-run] {tag}  {ticker:6s} {amount_desc}")
        return
    try:
        resp = client.submit_order(order)
        print(f"  {tag}  {ticker:6s} {amount_desc} -> status={resp.status} id={resp.id}")
        time.sleep(3)  # 3s delay after each order is sent to Alpaca
    except Exception as exc:
        print(f"  FAILED {tag}  {ticker:6s}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-generate", action="store_true",
        help="Skip running trading_model.py (reuse existing results/trades.csv)",
    )
    parser.add_argument(
        "--date", default=None,
        help="Trade a specific trade_date (YYYY-MM-DD) instead of the latest one",
    )
    parser.add_argument(
        "--learner-end", default=None,
        help="Learner end date (YYYY-MM-DD) passed to trading_model.py; defaults to the previous NYSE trading day",
    )
    parser.add_argument(
        "--no-refresh", action="store_true",
        help="Skip the network-bound preflight refresh when regenerating results/trades.csv (i.e. omit --refresh-data for trading_model.py)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without submitting orders (still reads account state)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Skip the wait for ORDER_TIME_ET (proceed even before the earliest order time)",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, TRADES_CSV)

    # Wait until ORDER_TIME_ET (ET) before proceeding with any trading actions.
    if args.force:
        print(">> --force given — skipping wait until ORDER_TIME_ET")
    elif args.dry_run:
        print(">> dry run — skipping wait until ORDER_TIME_ET (no orders will be submitted)")
    else:
        wait_until_order_time()

    current = datetime.now(ET)

    # Connect and pull read-only account/position state (also in dry-run for preview).
    client: Optional[Any] = None
    account = None
    positions = None
    try:
        client = make_client()
        account = client.get_account()
        positions = client.get_all_positions()
    except Exception as exc:
        if args.dry_run:
            print(f"  WARNING: could not fetch account state ({exc}); continuing dry run with defaults")
        else:
            sys.exit(f"ERROR: could not reach Alpaca API: {exc}")

    print_portfolio(account, positions)

    # 1) start_capital = current account equity.
    start_capital = acct_value(account, "equity") or DEFAULT_START_CAPITAL
    print(f">> start-capital={start_capital:,.2f}")

    # 2) Note and store any open positions on the account.
    open_positions: Dict[str, Any] = {}
    for p in positions or []:
        if isinstance(p, dict):
            open_positions[str(p.get("symbol"))] = p
        else:
            open_positions[str(getattr(p, "symbol"))] = p
    print(f">> Open positions noted/stored: {len(open_positions)}")

    # 3) Calculate today's trades using start_capital via trading_model.
    learner_end = args.learner_end or previous_trading_day(current)
    print(f">> learner-end={learner_end}")
    if not args.no_generate:
        run_generator(script_dir, learner_end, start_capital, refresh=not args.no_refresh)

    df = load_trades(csv_path)
    trades = select_trades(df, args.date)
    if trades.empty:
        sys.exit("ERROR: no trades to submit.")

    target_alloc = {
        str(t["ticker"]): float(t["alloc"])
        for _, t in trades.iterrows() if str(t["side"]).strip().lower() == "long"
    }
    print(f">> {len(trades)} trade(s) selected; {len(target_alloc)} long target allocation(s)")

    # 4) SELLS first (market orders).
    print(f">> Processing SELLS (dry_run={args.dry_run})")
    for ticker in sorted(open_positions):
        market_value = position_value(open_positions[ticker])
        if market_value <= 0:
            continue
        if ticker not in target_alloc:
            # Open position not in today's trades -> sell the ENTIRE position by qty.
            qty = position_qty(open_positions[ticker])
            if qty <= 0:
                continue
            print(f"  {ticker:6s} not in today's trades -> sell entire position (qty={qty:g})")
            submit_market(client, ticker, is_buy=False, qty=qty, dry_run=args.dry_run)
        else:
            delta = market_value - target_alloc[ticker]
            if delta > 0:
                # Over target -> sell the excess.
                submit_market(client, ticker, is_buy=False, notional=delta, dry_run=args.dry_run)
            else:
                print(f"  HOLD {ticker:6s} (market_value=${market_value:,.2f} <= alloc=${target_alloc[ticker]:,.2f})")

    # 5) BUYS next (market orders, notional).
    print(f">> Processing BUYS (dry_run={args.dry_run})")
    for ticker in sorted(target_alloc):
        alloc = target_alloc[ticker]
        if ticker not in open_positions:
            # In today's trades with no open position -> buy the entire alloc.
            print(f"  {ticker:6s} no open position -> buy full alloc")
            submit_market(client, ticker, is_buy=True, notional=alloc, dry_run=args.dry_run)
        else:
            delta = alloc - position_value(open_positions[ticker])
            if delta > 0:
                # Under target -> buy the shortfall.
                submit_market(client, ticker, is_buy=True, notional=delta, dry_run=args.dry_run)
            else:
                print(f"  HOLD {ticker:6s} (alloc=${alloc:,.2f} <= market_value=${position_value(open_positions[ticker]):,.2f})")

    if args.dry_run:
        print(">> Dry run complete — no orders were submitted.")
    else:
        print(">> Done. Review order statuses in your Alpaca paper dashboard.")


if __name__ == "__main__":
    main()


