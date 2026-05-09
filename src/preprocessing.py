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


def train_test_split_timewise(series, train_ratio=0.8):
    split = int(len(series) * train_ratio)
    return series[:split], series[split:]


def scale_train_test(train, test):
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()

    return train_scaled, test_scaled, scaler