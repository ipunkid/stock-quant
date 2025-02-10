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
    for window in [40, 60, 120, 250]:
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

def calculate_max_gain_this_year(df):
    if df.index[0] > YEAR_START_DATE:
        return None
    year_data = df[df.index >= YEAR_START_DATE]
    if not year_data.empty:
        year_start_price = year_data['close'].iloc[0]
    else:
        prev_year_data = df[df.index < YEAR_START_DATE]
        if prev_year_data.empty:
            return False
        year_start_price = prev_year_data['close'].iloc[-1]
    max_price_this_year = year_data['close'].max()
    return ((max_price_this_year - year_start_price) / year_start_price * 100).round(2)

def filter_criteria(df):
    latest = df.iloc[-1]
    # 最近30天(20个交易日)股价新高
    twenty_days_ago = datetime.now() - timedelta(days=30)
    year_high = df[(df.index >= YEAR_START_DATE) & (df.index <= twenty_days_ago)]["close"].max()
    recent_high_condition = df[df.index > twenty_days_ago]["close"].max() >= year_high
    # RPS120与RPS250之和大于185
    rps_condition = (latest["rps120"] + latest["rps250"]) > 190
    # 股价站上40日均线，且60日、120日、250日均线向上发散
    ma_condition = (latest["close"] > latest["ma40"]) and (latest["ma60"] > latest["ma120"]) and (latest["ma60"] > latest["ma250"])
    # 最近30天(20个交易日)最大跌幅小于30%
    current_price = latest["close"]
    max_price = df[df.index > twenty_days_ago]["close"].max()
    drawdown_condition = (max_price - current_price) / max_price <= 0.30
    # 最近一年涨幅不超过50%
    max_gain_this_year = calculate_max_gain_this_year(df)
    max_gain_condition = max_gain_this_year is not None and max_gain_this_year <= 80

    return all([recent_high_condition, rps_condition, ma_condition, drawdown_condition, max_gain_condition])

def process_stock(bs_code, all_stocks_data):
    hist_data = all_stocks_data[bs_code]
    if len(hist_data) >= 250:
        hist_data = calculate_moving_averages(hist_data)
        if filter_criteria(hist_data):
            latest = hist_data.iloc[-1]
            return {
                "code": bs_code,
                "rps50": round(latest["rps50"], 2),
                "rps120": round(latest["rps120"], 2),
                "rps250": round(latest["rps250"], 2),
                "max_yearly_return": calculate_max_gain_this_year(hist_data),
            }
    return None

def main():
    stock_list = [f.split(".")[1] for f in os.listdir(CACHE_DIR) if f.endswith(".json")]
    all_stocks_data = {bs_code: load_cache(bs_code) for bs_code in stock_list if load_cache(bs_code) is not None and len(load_cache(bs_code)) >= 250}

    for period in [50, 120, 250]:
        all_stocks_data = calculate_rps(all_stocks_data, period)

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(process_stock, bs_code, all_stocks_data) for bs_code in all_stocks_data.keys()]
        selected_stocks = [result for future in as_completed(futures) if (result := future.result()) is not None]

    selected_stocks_df = pd.DataFrame(selected_stocks)
    timestamp = datetime.today().strftime("%y%m%d")
    selected_stocks_df.to_csv(f"rps_first_selected_stocks_{timestamp}.csv", index=False)

    print("\nSelected Stocks:")
    print(tabulate(selected_stocks_df, headers="keys", tablefmt="grid", numalign="right"))

if __name__ == "__main__":
    main()
