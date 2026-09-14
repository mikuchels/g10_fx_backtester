"""
Phase 2 — Strategy Signal Generators.

Each function takes market data and returns a DataFrame of daily position
weights (columns = currencies). Positive = long, negative = short.

Strategies:
  1. Carry     — go long high-yielding currencies, short low-yielding
  2. Momentum  — time-series momentum across multiple lookback windows
  3. Mean Reversion — fade short-term z-score extremes
"""

import pandas as pd
import numpy as np

from config import (
    CARRY_LONG_N, CARRY_SHORT_N,
    MOMENTUM_LOOKBACKS,
    MR_WINDOW, MR_ZSCORE_THRESHOLD,
)


def carry_signals(carry_df, long_n=CARRY_LONG_N, short_n=CARRY_SHORT_N):
    """
    Classic FX carry trade.

    At each month-end, rank all G10 currencies by their interest rate
    differential vs USD. Go long the top `long_n` currencies (highest
    carry) with equal weight +1/long_n each. Go short the bottom
    `short_n` currencies (lowest carry) with equal weight -1/short_n each.

    Positions are held constant within each month (rebalance monthly).
    Dollar-neutral when long_n == short_n.

    Parameters
    ----------
    carry_df : DataFrame  — daily carry differentials (annualised, decimal)
    long_n   : int        — number of currencies to go long
    short_n  : int        — number of currencies to go short

    Returns
    -------
    DataFrame of daily position weights, same shape as carry_df.
    """
    carry_monthly = carry_df.resample("ME").last().dropna(how="all")
    signals_m = pd.DataFrame(0.0, index=carry_monthly.index, columns=carry_monthly.columns)

    for date in carry_monthly.index:
        row = carry_monthly.loc[date].dropna().sort_values(ascending=False)
        if len(row) < long_n + short_n:
            continue

        longs = row.index[:long_n]
        shorts = row.index[-short_n:]
        signals_m.loc[date, longs] = 1.0 / long_n
        signals_m.loc[date, shorts] = -1.0 / short_n

    # Forward-fill monthly signals to daily frequency
    signals = signals_m.reindex(carry_df.index).ffill().fillna(0.0)
    return signals


def momentum_signals(returns_df, lookbacks=None):
    """
    Time-series momentum (TSMOM), following Moskowitz, Ooi & Pedersen (2012).

    For each currency, compute the cumulative return over each lookback
    window. The signal is the sign of the cumulative return (+1 or -1),
    averaged across all lookback windows. Signals are then cross-sectionally
    normalised so that absolute weights sum to 1 each day.

    Parameters
    ----------
    returns_df : DataFrame  — daily spot returns per currency
    lookbacks  : list[int]  — lookback windows in trading days

    Returns
    -------
    DataFrame of daily position weights.
    """
    if lookbacks is None:
        lookbacks = MOMENTUM_LOOKBACKS

    combined = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)

    for lb in lookbacks:
        cum_ret = returns_df.rolling(window=lb, min_periods=lb).sum()
        combined += np.sign(cum_ret) / len(lookbacks)

    # Cross-sectional normalisation: |weights| sum to 1
    abs_sum = combined.abs().sum(axis=1).replace(0, np.nan)
    signals = combined.div(abs_sum, axis=0).fillna(0.0)
    return signals


def mean_reversion_signals(returns_df, window=MR_WINDOW, z_threshold=MR_ZSCORE_THRESHOLD):
    """
    Short-term mean reversion strategy.

    Compute a rolling z-score of daily returns. When the z-score exceeds
    the threshold in either direction, fade the move:
      - z > +threshold  → short (overbought)
      - z < -threshold  → long  (oversold)

    Signals normalised so absolute weights sum to 1.

    Parameters
    ----------
    returns_df  : DataFrame — daily spot returns per currency
    window      : int       — rolling window for z-score (trading days)
    z_threshold : float     — signal trigger threshold

    Returns
    -------
    DataFrame of daily position weights.
    """
    rolling_mean = returns_df.rolling(window=window, min_periods=window).mean()
    rolling_std = returns_df.rolling(window=window, min_periods=window).std()
    z_scores = (returns_df - rolling_mean) / rolling_std.replace(0, np.nan)

    signals = pd.DataFrame(0.0, index=returns_df.index, columns=returns_df.columns)
    signals[z_scores > z_threshold] = -1.0   # overbought → short
    signals[z_scores < -z_threshold] = 1.0   # oversold   → long

    abs_sum = signals.abs().sum(axis=1).replace(0, np.nan)
    signals = signals.div(abs_sum, axis=0).fillna(0.0)
    return signals