"""
Phase 1 — Data Infrastructure.

Downloads and caches:
  - Daily FX spot rates from Yahoo Finance (9 G10 currency pairs vs USD)
  - Short-term interest rates from FRED (10 currencies including USD)

All data is cached locally as Parquet files so subsequent runs are instant.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred

from config import (
    G10_CURRENCIES, FX_TICKERS, INVERT, FRED_RATES,
    DEFAULT_START, DEFAULT_END, DAYS_PER_YEAR,
)

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ------------------------------------------------------------------
# FX spot data
# ------------------------------------------------------------------

def fetch_fx_spots(start=DEFAULT_START, end=DEFAULT_END, use_cache=True):
    """
    Download daily FX spot closing prices from Yahoo Finance.

    Returns a DataFrame indexed by date with one column per G10 currency.
    All rates are expressed in XXXUSD convention (value of 1 unit of foreign
    currency in USD). A rising value means the foreign currency is appreciating.
    """
    cache_path = os.path.join(CACHE_DIR, "fx_spots.parquet")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if not df.empty:
            return df

    tickers = list(FX_TICKERS.values())
    ticker_to_ccy = {v: k for k, v in FX_TICKERS.items()}

    print(f"Downloading FX spot data for {len(tickers)} pairs...")
    raw = yf.download(tickers, start=start, end=end, progress=True)

    # yfinance >= 0.2.31 returns multi-level columns: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    # Build output: rename tickers -> currency codes, invert where needed
    spots = pd.DataFrame(index=close.index)
    for ticker, ccy in ticker_to_ccy.items():
        if ticker in close.columns:
            s = close[ticker].copy()
            if INVERT[ccy]:
                s = 1.0 / s
            spots[ccy] = s
        else:
            print(f"  WARNING: No data returned for {ccy} ({ticker})")

    spots = spots.sort_index().dropna(how="all")
    spots.to_parquet(cache_path)
    print(f"  ✓ {len(spots)} trading days, {len(spots.columns)} currencies loaded.")
    return spots


# ------------------------------------------------------------------
# Interest rate data
# ------------------------------------------------------------------

def fetch_interest_rates(fred_api_key, start=DEFAULT_START, end=DEFAULT_END, use_cache=True):
    """
    Download short-term interest rates from FRED.

    USD uses the daily 3-month T-bill rate (DTB3).  Other currencies use
    monthly OECD 3-month interbank rates, which are forward-filled to daily.

    Returns a DataFrame indexed by business date, values in decimal
    (e.g. 0.05 = 5% p.a.).
    """
    cache_path = os.path.join(CACHE_DIR, "interest_rates.parquet")
    if use_cache and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if not df.empty:
            return df

    fred = Fred(api_key=fred_api_key)
    rates = {}

    for ccy, series_id in FRED_RATES.items():
        print(f"  Fetching {ccy} rate ({series_id})...")
        try:
            s = fred.get_series(series_id, observation_start=start, observation_end=end)
            s = s.dropna()
            if not s.empty:
                rates[ccy] = s / 100.0  # percentage -> decimal
        except Exception as e:
            print(f"    WARNING: {ccy} ({series_id}) failed: {e}")

    rates_df = pd.DataFrame(rates)
    rates_df.index = pd.to_datetime(rates_df.index)
    rates_df = rates_df.sort_index()

    # Forward-fill monthly OECD data to business-day frequency
    bday_idx = pd.bdate_range(start=rates_df.index.min(), end=rates_df.index.max())
    rates_df = rates_df.reindex(bday_idx).ffill().bfill()

    rates_df.to_parquet(cache_path)
    print(f"  ✓ Rates loaded for {len(rates_df.columns)} currencies, {len(rates_df)} days.")
    return rates_df


# ------------------------------------------------------------------
# Derived quantities
# ------------------------------------------------------------------

def compute_spot_returns(spots_df):
    """Daily simple returns from spot levels."""
    return spots_df.pct_change().dropna(how="all")


def compute_carry_differential(rates_df):
    """
    Interest rate differential for each G10 currency vs USD.
    Positive = foreign rate > USD rate = positive carry when long the currency.
    """
    if "USD" not in rates_df.columns:
        raise ValueError("USD rate column not found.")

    carry = pd.DataFrame(index=rates_df.index)
    for ccy in G10_CURRENCIES:
        if ccy in rates_df.columns:
            carry[ccy] = rates_df[ccy] - rates_df["USD"]
    return carry


# ------------------------------------------------------------------
# Master loader
# ------------------------------------------------------------------

def load_all_data(fred_api_key, start=DEFAULT_START, end=DEFAULT_END, use_cache=True):
    """
    Load and align all data into a single dict.

    Returns
    -------
    dict with keys: 'spots', 'returns', 'rates', 'carry'
        All DataFrames share a common date index and currency columns.
    """
    spots = fetch_fx_spots(start, end, use_cache)
    returns = compute_spot_returns(spots)
    rates = fetch_interest_rates(fred_api_key, start, end, use_cache)
    carry = compute_carry_differential(rates)

    # Align on common dates and currencies
    common_dates = returns.index.intersection(carry.index)
    common_ccys = sorted(set(returns.columns) & set(carry.columns))

    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between FX spot and rate data.")
    if len(common_ccys) == 0:
        raise ValueError("No overlapping currencies between FX spot and rate data.")

    data = {
        "spots": spots.reindex(index=common_dates, columns=common_ccys),
        "returns": returns.reindex(index=common_dates, columns=common_ccys).fillna(0.0),
        "rates": rates.reindex(index=common_dates),
        "carry": carry.reindex(index=common_dates, columns=common_ccys).fillna(0.0),
    }

    print(
        f"\n✓ Aligned dataset: {len(common_dates)} days, {len(common_ccys)} currencies "
        f"({common_dates[0].date()} → {common_dates[-1].date()})"
    )
    print(f"  Currencies: {', '.join(common_ccys)}")
    return data