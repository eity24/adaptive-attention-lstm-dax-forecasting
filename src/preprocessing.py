import numpy as np
from sklearn.preprocessing import StandardScaler


def keep_required_columns(df):
    date_col = "Date"

    close_col = None
    for col in df.columns:
        if str(col).lower() == "close" or "close" in str(col).lower():
            close_col = col
            break

    if close_col is None:
        raise ValueError(f"No close column found. Available columns: {list(df.columns)}")

    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    return df[[date_col, "Close"]].copy()


def compute_log_returns(df):
    df = df.copy()
    df["log_return"] = np.log(df["Close"]).diff()
    df = df.dropna().reset_index(drop=True)
    return df

def compute_rolling_volatility(df, window=20):

    df["rolling_volatility"] = (
        df["log_return"]
        .rolling(window=window)
        .std()
    )

    df = df.dropna()

    return df

def compute_technical_indicators(df):
    df = df.copy()

    # 20-day moving average of log returns
    df["rolling_mean_20"] = (
        df["log_return"]
        .rolling(window=20)
        .mean()
    )

    # RSI calculation based on log returns
    delta = df["log_return"]

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / (avg_loss + 1e-8)

    df["rsi_14"] = 100 - (100 / (1 + rs))

    df = df.dropna()

    return df

def keep_ohlc_columns(df):
    df = df.copy()

    date_col = "Date"

    required_cols = {}

    for col in df.columns:
        col_lower = str(col).lower()

        if col_lower == "open" or "open" in col_lower:
            required_cols[col] = "Open"

        elif col_lower == "high" or "high" in col_lower:
            required_cols[col] = "High"

        elif col_lower == "low" or "low" in col_lower:
            required_cols[col] = "Low"

        elif col_lower == "close" or "close" in col_lower:
            required_cols[col] = "Close"

    df = df.rename(columns=required_cols)

    return df[[date_col, "Open", "High", "Low", "Close"]].copy()


def train_test_split_timewise(series, train_ratio=0.8):
    split = int(len(series) * train_ratio)
    return series[:split], series[split:]


def scale_train_test(train, test):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

    return train_scaled, test_scaled, scaler