import os
import yfinance as yf
import pandas as pd


def download_dax_data(start_date="2010-01-01", end_date="2026-04-02", save_path="data/raw/dax.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = yf.download("^GDAXI", start=start_date, end=end_date, auto_adjust=False)

    if df.empty:
        raise ValueError("No data downloaded")

    # index কে normal column বানাও
    df = df.reset_index()

    # যদি multi-level column হয়, flatten করো
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_")
            for col in df.columns
        ]

    # Date column clean করা
    date_col = None
    for col in df.columns:
        if "date" in str(col).lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"No date column found. Columns are: {list(df.columns)}")

    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    df.to_csv(save_path, index=False)
    return df


def load_dax_data(file_path="data/raw/dax.csv"):
    df = pd.read_csv(file_path)

    # column names clean করা
    df.columns = [str(col).strip() for col in df.columns]

    # Date column খোঁজা
    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError(f"'Date' column not found. Available columns: {list(df.columns)}")

    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    return df