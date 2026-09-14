"""
Configuration and constants for G10 FX Strategy Backtester.
"""

# G10 currencies (excluding USD, which serves as the funding/base currency)
G10_CURRENCIES = ["EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"]

# yfinance tickers for each currency vs USD
FX_TICKERS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "JPY": "USDJPY=X",
    "CHF": "USDCHF=X",
    "AUD": "AUDUSD=X",
    "NZD": "NZDUSD=X",
    "CAD": "USDCAD=X",
    "SEK": "USDSEK=X",
    "NOK": "USDNOK=X",
}

# True = quoted as USDXXX on yfinance, so we invert to get "foreign ccy per 1 USD" -> XXXUSD
# After inversion, a rising value = foreign currency appreciation vs USD
INVERT = {
    "EUR": False,  # EURUSD: EUR up = rate up, no inversion
    "GBP": False,  # GBPUSD: GBP up = rate up, no inversion
    "JPY": True,   # USDJPY: JPY up = rate down, invert
    "CHF": True,   # USDCHF: CHF up = rate down, invert
    "AUD": False,  # AUDUSD: AUD up = rate up, no inversion
    "NZD": False,  # NZDUSD: NZD up = rate up, no inversion
    "CAD": True,   # USDCAD: CAD up = rate down, invert
    "SEK": True,   # USDSEK: SEK up = rate down, invert
    "NOK": True,   # USDNOK: NOK up = rate down, invert
}

# FRED series IDs for short-term interest rates
# USD uses the daily 3-month T-bill; others use monthly OECD 3-month interbank rates
FRED_RATES = {
    "USD": "DTB3",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "CHF": "IR3TIB01CHM156N",
    "AUD": "IR3TIB01AUM156N",
    "NZD": "IR3TIB01NZM156N",
    "CAD": "IR3TIB01CAM156N",
    "SEK": "IR3TIB01SEM156N",
    "NOK": "IR3TIB01NOM156N",
}

# One-way transaction costs in decimal (1 bp = 0.0001)
# Majors are tighter; Scandies are wider
TRANSACTION_COSTS = {
    "EUR": 0.00010,
    "GBP": 0.00015,
    "JPY": 0.00010,
    "CHF": 0.00020,
    "AUD": 0.00020,
    "NZD": 0.00030,
    "CAD": 0.00020,
    "SEK": 0.00040,
    "NOK": 0.00040,
}

# ---- Default parameters ----
DEFAULT_START = "2005-01-01"
DEFAULT_END = "2024-12-31"
DAYS_PER_YEAR = 252

# Carry
CARRY_LONG_N = 3
CARRY_SHORT_N = 3

# Momentum
MOMENTUM_LOOKBACKS = [21, 63, 252]  # 1 month, 3 months, 12 months (trading days)

# Mean reversion
MR_WINDOW = 15
MR_ZSCORE_THRESHOLD = 1.5

# Risk
TARGET_VOLATILITY = 0.10  # 10% annualised per position
VOL_LOOKBACK = 63         # ~3 months for realised vol estimation
VOL_LEVERAGE_CAP = 5.0    # max leverage from vol targeting