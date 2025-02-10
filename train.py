#!/usr/bin/env python3

import pandas as pd
from datetime import datetime, timedelta
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tabulate import tabulate

# 缓存目录
CACHE_DIR = "stock_cache"
# 当前年份
CURRENT_YEAR = datetime.now().year
YEAR_START_DATE = datetime(CURRENT_YEAR, 1, 1)

def load_cache(bs_code):
    for prefix in ['sh', 'sz']:
        cache_file = os.path.join(CACHE_DIR, f"{prefix}.{bs_code}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached_data = json.load(f)
                df = pd.DataFrame(cached_data)
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                return df.sort_index()
            except (json.JSONDecodeError, ValueError):
                print(f"Error reading cache for {bs_code}.")
    print(f"No cache file found for {bs_code}")
    return None

def calculate_moving_averages(df):
    for window in [10, 20, 200, 250]:
        df[f"ma{window}"] = df["close"].rolling(window=window).mean()
    return df

def calculate_rps(all_stocks_data, period):
    all_closes = pd.DataFrame({code: data['close'] for code, data in all_stocks_data.items()})
    pct_changes = all_closes.pct_change(periods=period, fill_method=None)
    ranks = pct_changes.iloc[-1].rank(pct=True)
    rps = ranks * 100
    for code in all_stocks_data.keys():
        all_stocks_data[code][f"rps{period}"] = rps[code]
    return all_stocks_data

def filter_criteria(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]

    def check_rps():
        return (latest["rps120"] + latest["rps250"]) > 185

    def check_drawdown():
        twenty_days_ago = datetime.now() - timedelta(days=20)
        current_price = latest["close"]
        max_price = df[df.index > twenty_days_ago]["close"].max()
        return (max_price - current_price) / max_price <= 0.25

    def check_ma_crossover(days, short_ma, long_ma, threshold):
        last_n_days = df.iloc[-days:]
        condition = (last_n_days['close'] > last_n_days[f'ma{short_ma}']) & (last_n_days['close'] > last_n_days[f'ma{long_ma}'])
        return condition.sum() >= threshold

    def check_price_to_year_high():
        one_year_ago = datetime.now() - timedelta(days=365)
        year_high = df[df.index >= one_year_ago]["close"].max()
        return latest["close"] >= 0.8 * year_high

    def check_ma_trend():
        last_5_days = df.iloc[-5:]
        ma_condition_5days = ((last_5_days['ma20'].diff() > 0) & (last_5_days['ma10'] > last_5_days['ma20'])).all()
        ma_condition_latest = (
            (latest['ma10'] > previous['ma10']) and
            (latest['ma20'] > previous['ma20']) and
            (latest['ma10'] > latest['ma20'])
        )
        return ma_condition_5days or ma_condition_latest

    def check_price_above_ma20():
        return latest['close'] > latest['ma20']

    def check_ytd_increase():
        # Try to get data from current year
        current_year_data = df[df.index >= YEAR_START_DATE]
        if not current_year_data.empty:
            year_start_price = current_year_data["close"].iloc[0]
        else:
            # If no data for current year, get last price from previous year
            prev_year_data = df[df.index < YEAR_START_DATE]
            if prev_year_data.empty:
                return False  # Not enough data to make a decision
            year_start_price = prev_year_data["close"].iloc[-1]
        
        return (latest["close"] - year_start_price) / year_start_price <= 0.6

    conditions = [
        check_rps(),
        check_drawdown(),
        check_ma_crossover(30, 200, 250, 25),
        check_ma_crossover(10, 20, 20, 9),
        check_ma_crossover(4, 10, 20, 3),
        check_price_to_year_high(),
        check_ma_trend(),
        check_price_above_ma20(),
        check_ytd_increase()
    ]

    return all(conditions)

def process_stock(bs_code, all_stocks_data):
    hist_data = all_stocks_data[bs_code]
    if len(hist_data) >= 250:
        hist_data = calculate_moving_averages(hist_data)
        if filter_criteria(hist_data):
            latest = hist_data.iloc[-1]
            return {
                "code": bs_code,
                "rps120": round(latest["rps120"], 2),
                "rps250": round(latest["rps250"], 2),
            }
    return None

def main():
    stock_list = [f.split(".")[1] for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    all_stocks_data = {bs_code: load_cache(bs_code) for bs_code in stock_list if load_cache(bs_code) is not None and len(load_cache(bs_code)) >= 250}

    for period in [120, 250]:
        all_stocks_data = calculate_rps(all_stocks_data, period)

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_stock, bs_code, all_stocks_data) for bs_code in all_stocks_data.keys()]
        selected_stocks = [result for future in as_completed(futures) if (result := future.result()) is not None]

    selected_stocks_df = pd.DataFrame(selected_stocks)
    timestamp = datetime.today().strftime("%y%m%d")
    selected_stocks_df.to_csv(f"train_track_selected_stocks_{timestamp}.csv", index=False)

    print("\nSelected Stocks:")
    print(tabulate(selected_stocks_df, headers="keys", tablefmt="grid", numalign="right"))

if __name__ == "__main__":
    main()
