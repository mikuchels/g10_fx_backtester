"""
Phase 3 — Backtesting Engine
Phase 4 — Performance Analytics
Phase 5 — Strategy Combination and Correlation

Vectorised backtest (no for-loops over dates). Includes:
  - 1-day signal lag to prevent look-ahead bias
  - Per-position volatility targeting
  - Realistic transaction cost deduction on position changes
  - Carry income accrual on all positions
  - Full tear sheet: Sharpe, drawdown, skewness, hit rate, etc.
  - Multi-strategy combination with equal or inverse-vol weighting
"""

import pandas as pd
import numpy as np

from config import (
    DAYS_PER_YEAR, TRANSACTION_COSTS,
    TARGET_VOLATILITY, VOL_LOOKBACK, VOL_LEVERAGE_CAP,
)


# ==================================================================
# Phase 3 — Backtesting Engine
# ==================================================================

def _vol_target_positions(positions, returns_df, target_vol, lookback, leverage_cap):
    """Scale each currency position by target_vol / realised_vol (lagged)."""
    ann_vol = returns_df.rolling(window=lookback, min_periods=lookback).std() * np.sqrt(DAYS_PER_YEAR)
    scalar = target_vol / ann_vol.replace(0, np.nan)
    scalar = scalar.clip(upper=leverage_cap).shift(1)  # lag to avoid look-ahead
    return (positions * scalar).fillna(0.0)


def _turnover_costs(positions, tc_map):
    """Daily transaction cost from absolute position changes."""
    delta = positions.diff().abs().fillna(0.0)
    costs = pd.DataFrame(0.0, index=positions.index, columns=positions.columns)
    for ccy in positions.columns:
        costs[ccy] = delta[ccy] * tc_map.get(ccy, 0.0002)
    return costs.sum(axis=1)


def run_backtest(
    signals,
    returns_df,
    carry_df,
    apply_tc=True,
    apply_vol_target=True,
    target_vol=TARGET_VOLATILITY,
    vol_lookback=VOL_LOOKBACK,
    leverage_cap=VOL_LEVERAGE_CAP,
    tc_map=None,
):
    """
    Run a vectorised backtest for a single strategy.

    Signal on day T is applied to the return on day T+1 (1-day execution lag).
    Total return per position = spot return + daily carry accrual.

    Parameters
    ----------
    signals          : DataFrame — raw strategy weights per currency per day
    returns_df       : DataFrame — daily spot returns per currency
    carry_df         : DataFrame — annualised carry differentials per currency
    apply_tc         : bool      — deduct transaction costs
    apply_vol_target : bool      — apply per-position volatility targeting
    target_vol       : float     — annualised vol target (decimal)
    vol_lookback     : int       — lookback for realised vol
    leverage_cap     : float     — max leverage from vol targeting
    tc_map           : dict      — override transaction cost map

    Returns
    -------
    dict with:
        positions       — held positions (after lag and vol scaling)
        pnl_by_ccy      — daily PnL contribution per currency
        gross_returns    — daily portfolio return before TC
        net_returns      — daily portfolio return after TC
        tc_daily         — daily transaction cost drag
        equity_gross     — cumulative NAV (gross)
        equity_net       — cumulative NAV (net)
    """
    if tc_map is None:
        tc_map = TRANSACTION_COSTS

    # 1-day lag: signal on T, position on T+1
    positions = signals.shift(1).fillna(0.0)

    # Volatility targeting (per position)
    if apply_vol_target:
        positions = _vol_target_positions(positions, returns_df, target_vol, vol_lookback, leverage_cap)

    # Total return = spot return + carry accrual
    daily_carry = carry_df.reindex_like(returns_df).fillna(0.0) / DAYS_PER_YEAR
    total_return_per_ccy = returns_df + daily_carry

    # Portfolio PnL
    pnl_by_ccy = positions * total_return_per_ccy
    gross_daily = pnl_by_ccy.sum(axis=1)

    # Transaction costs
    if apply_tc:
        tc_daily = _turnover_costs(positions, tc_map)
    else:
        tc_daily = pd.Series(0.0, index=returns_df.index)

    net_daily = gross_daily - tc_daily

    return {
        "positions": positions,
        "pnl_by_ccy": pnl_by_ccy,
        "gross_returns": gross_daily,
        "net_returns": net_daily,
        "tc_daily": tc_daily,
        "equity_gross": (1.0 + gross_daily).cumprod(),
        "equity_net": (1.0 + net_daily).cumprod(),
    }


# ==================================================================
# Phase 4 — Performance Analytics
# ==================================================================

def drawdown_series(equity):
    """Drawdown series from an equity curve (values <= 0)."""
    return equity / equity.cummax() - 1.0


def _max_drawdown_duration(equity):
    """Longest drawdown duration in trading days."""
    peak = equity.cummax()
    in_dd = equity < peak
    if not in_dd.any():
        return 0
    groups = (~in_dd).cumsum()
    dd_groups = groups[in_dd]
    if dd_groups.empty:
        return 0
    return dd_groups.groupby(dd_groups).count().max()


def performance_table(returns, name="Strategy"):
    """
    Compute a full set of performance metrics from daily returns.
    Returns a dict suitable for a single row in a summary table.
    """
    r = returns.dropna()
    if r.empty:
        return {"Name": name, "Error": "No data"}

    n_days = len(r)
    n_years = n_days / DAYS_PER_YEAR

    # Return
    total_return = (1 + r).prod() - 1
    ann_return_geom = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_return_arith = r.mean() * DAYS_PER_YEAR

    # Risk
    ann_vol = r.std() * np.sqrt(DAYS_PER_YEAR)
    sharpe = ann_return_arith / ann_vol if ann_vol > 0 else 0.0

    # Drawdown
    eq = (1 + r).cumprod()
    dd = drawdown_series(eq)
    max_dd = dd.min()
    max_dd_dur = _max_drawdown_duration(eq)
    calmar = abs(ann_return_geom / max_dd) if max_dd != 0 else 0.0

    # Distribution
    skew = r.skew()
    kurt = r.kurtosis()

    # Monthly hit rate
    monthly = r.resample("ME").sum()
    hit_rate = (monthly > 0).mean() if len(monthly) > 0 else 0.0

    return {
        "Name": name,
        "Total Return": f"{total_return:.1%}",
        "Ann. Return": f"{ann_return_geom:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": f"{max_dd:.1%}",
        "Max DD Duration": f"{int(max_dd_dur)}d",
        "Calmar Ratio": round(calmar, 2),
        "Skewness": round(skew, 2),
        "Excess Kurtosis": round(kurt, 2),
        "Monthly Hit Rate": f"{hit_rate:.0%}",
        "Best Month": f"{monthly.max():.2%}" if len(monthly) > 0 else "N/A",
        "Worst Month": f"{monthly.min():.2%}" if len(monthly) > 0 else "N/A",
    }


def monthly_returns_matrix(returns):
    """Pivot daily returns into a Year × Month matrix for heatmap display."""
    monthly = returns.resample("ME").sum()
    df = pd.DataFrame(
        {"Year": monthly.index.year, "Month": monthly.index.month, "Return": monthly.values}
    )
    pivot = df.pivot_table(values="Return", index="Year", columns="Month", aggfunc="sum")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    pivot["Annual"] = pivot.sum(axis=1)
    return pivot


# ==================================================================
# Phase 5 — Strategy Combination
# ==================================================================

def strategy_correlation(results_dict):
    """Correlation matrix of daily net returns across strategies."""
    rets = pd.DataFrame({name: r["net_returns"] for name, r in results_dict.items()})
    return rets.corr()


def combine_strategies(results_dict, method="equal", vol_lookback=63):
    """
    Combine multiple strategy return streams into a single portfolio.

    Parameters
    ----------
    results_dict : dict[str, backtest_result]
    method       : 'equal' — equal weight
                   'inverse_vol' — weight inversely by rolling volatility
    vol_lookback : int — lookback for vol estimation (inverse_vol only)

    Returns
    -------
    dict with: 'net_returns', 'equity_net', 'weights', 'component_returns'
    """
    comp = pd.DataFrame({name: r["net_returns"] for name, r in results_dict.items()}).dropna()

    if method == "equal":
        combined = comp.mean(axis=1)
        weights = pd.Series(1.0 / len(comp.columns), index=comp.columns)

    elif method == "inverse_vol":
        rolling_vol = comp.rolling(vol_lookback).std()
        inv_vol = 1.0 / rolling_vol.replace(0, np.nan)
        w = inv_vol.div(inv_vol.sum(axis=1), axis=0).shift(1)
        w = w.fillna(1.0 / len(comp.columns))
        combined = (comp * w).sum(axis=1)
        weights = w.mean()
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "net_returns": combined,
        "equity_net": (1 + combined).cumprod(),
        "weights": weights,
        "component_returns": comp,
    }