import numpy as np
import pandas as pd
import os
import glob
import pickle
import json
import sys
import argparse
from datetime import datetime, timedelta
from scipy import stats
from openbb_terminal.sdk import openbb

def get_ar_dates(symbol, config_dict):
    '''
    Returns: The annual report dates for each symbol.

    目的：獲取特定股票符號的年報發布日期。
    流程：
    - 使用 os.path.join 和 glob.glob 結合 symbol 和 config_dict 中的路徑來定位年報文件夾。
    - 只選擇路徑下的文件夾（忽略文件），將每個文件夾名視為一個年報日期。
    - 將這些日期排序後返回。
    '''
    symbol_path = os.path.join(config_dict['annual_reports_pdf_save_directory'], symbol)
    folder_names = [os.path.basename(folder) for folder in glob.glob(os.path.join(symbol_path, '*')) \
                        if os.path.isdir(folder)]
    return sorted(folder_names)

def get_pct_returns_defined_date(price_data, start_date, end_date, tolerance_days = 7):
    '''
    Function to give percentage returns over a defined time range.
    Args:
        price_data: A pandas DF containing the overall price history of a stock symbol
        start_date: The start date of computing pct returns
        end_date: The end date of computing pct returns
        tolerance_days: If the price data is missing for more than this then return nan.
                        Some data can be missing due to Weekends or Holidays
    Returns:
        Percentage price change between start and end date.

    目的：計算從 start_date 到 end_date 期間的股票價格百分比變化。
    流程：
    - 先篩選出位於起止日期範圍內的價格數據。
    - 檢查數據開始和結束日期與所需日期的差異是否超過容忍天數。如果是，返回 np.nan。
    - 計算指定日期範圍內的價格百分比變化。
    '''
    price_data_range = price_data.sort_index().loc[lambda x: (x.index>=start_date) & \
                                                               (x.index <= end_date)]
    num_days_diff_start = (price_data_range.index[0] - start_date).days
    num_days_diff_end = (end_date - price_data_range.index[-1]).days
    if ((num_days_diff_start > tolerance_days) | (num_days_diff_end >= tolerance_days)):
        return np.nan
    start_price = price_data_range.iloc[0]['Adj Close']
    end_price = price_data_range.iloc[-1]['Adj Close']
    price_pct_diff = ((end_price - start_price)/start_price) * 100.0
    return price_pct_diff

def get_pct_returns_range(price_data, start_date, end_date, quantile, tolerance_days = 7):
    '''
    Function to give quantile based percentage returns between a defined time range.
    For example to get the maximum returns achieved from start_date and before the end_date
    Args:
        price_data: A pandas DF containing the overall price history of a stock symbol
        start_date: The start date of computing pct returns
        end_date: The end date of computing pct returns
        tolerance_days: If the price data is missing for more than this then return nan.
                        Some data can be missing due to Weekends or Holidays
    Returns:
        Specified quantile of percentage price change between start_date and before the end_date

    目的：獲取定義時間範圍內特定百分位的價格百分比變化。
    流程：
    - 與 get_pct_returns_defined_date 類似，但是在計算百分比變化時，會根據指定的百分位數（quantile）來計算結束價格（如最大值或最小值）。
    '''
    price_data_range = price_data.sort_index().loc[lambda x: (x.index>=start_date) & \
                                                               (x.index <= end_date)]
    num_days_diff_start = (price_data_range.index[0] - start_date).days
    num_days_diff_end = (end_date - price_data_range.index[-1]).days
    if ((num_days_diff_start > tolerance_days) | (num_days_diff_end >= tolerance_days)):
        return np.nan
    start_price = price_data_range.iloc[0]['Adj Close']
    end_price = price_data_range['Adj Close'].quantile(quantile)
    price_pct_diff = ((end_price - start_price)/start_price) * 100.0
    return price_pct_diff

def get_all_targets(price_data, start_date, num_days_12m, prepend_string):
    '''
    Function to return a dictionary of targets which contain percentage returns over different 
    time ranges and quantiles
    Args:
        price_data: A pandas DF containing the overall price history of a stock symbol
        start_date: The start date of computing pct returns
        num_days_12m: Num of Days between start_date and successive annual report date
        prepend_string: Denoting the category of price_data. One of 'target' and 'sp500'
        
    目的：生成不同時間範圍和百分位數的目標回報。
    流程：
    - 對多個時間段（3, 6, 9, 12個月）和百分位數（最大和最小）使用前面的函數來計算回報。
    - 這些回報被保存在一個字典中，並附加一個前綴字符串（如 'target' 或 'sp500'）。
    '''
    target_returns_dict = {}
    try:
        #98th percentile is proxy for max returns
        target_returns_dict['{}_max'.format(prepend_string)] = get_pct_returns_range(price_data, 
                                                                  start_date, 
                                                                  start_date + timedelta(days=num_days_12m), 
                                                                  0.98)
        #2nd percentile is proxy for min returns
        target_returns_dict['{}_min'.format(prepend_string)] = get_pct_returns_range(price_data, 
                                                                  start_date, 
                                                                  start_date + timedelta(days=num_days_12m), 
                                                                  0.02)
    except:
        target_returns_dict['{}_max'.format(prepend_string)] = np.nan
        target_returns_dict['{}_min'.format(prepend_string)] = np.nan

    #Get returns for 3, 6, 9 and 12 month duration. Some alteration is done in time duration
    #as the annual reports may be release in an interval of less or more than 12 months.
    for period in [3, 6, 9, 12]:
        num_days = int(num_days_12m * (period/12))
        end_date = start_date + timedelta(days=num_days)
        try:
            pct_returns = get_pct_returns_defined_date(price_data, start_date, end_date)
        except:
            pct_returns = np.nan
        target_returns_dict['{}_{}m'.format(prepend_string, str(period))] = pct_returns
    return target_returns_dict

def make_targets(symbol, start_date, end_date, price_data_sp500, config_dict):
    '''
    Function to generate target return information for each symbol based on 
    annual report dates
    Args:
        symbol: stock ticker
        start_date: overall historical start date from where price data is to be fetched
        end_date: overall end date upto which price data is to be fetched
        price_data_sp500: Prefetched dataframe for ticker ^GSPC which gives price data for S&P500
    Returns:
        Pandas DF containing percentage returns between annual report dates for the symbol

    目的：為每個股票符號創建目標回報信息。
    流程：
    - 使用 openbb.stocks.load 從股市數據庫中加載特定股票的價格數據。
    - 遍歷每個年報日期，使用 get_all_targets 為每個報告期間生成目標回報數據。
    - 同時計算股票和S&P 500的目標回報，並將這些數據合併到一個 Pandas 數據框中。
    '''
    price_data = openbb.stocks.load(symbol, start_date=start_date, end_date=end_date, 
                                    verbose=False)
    ar_dates = get_ar_dates(symbol, config_dict)
    df = pd.DataFrame()
    for i in range(len(ar_dates)-1):
        curr_report_date = datetime.strptime(ar_dates[i], '%Y-%m-%d')
        #Start and end dates are offset by 2 days to be conservative and allowing the price to settle.
        curr_start_date = datetime.strptime(ar_dates[i], '%Y-%m-%d') + timedelta(days=2)
        curr_end_date_12m = datetime.strptime(ar_dates[i+1], '%Y-%m-%d') - timedelta(days=2)
        num_days_12m = (curr_end_date_12m - curr_start_date).days
        if (num_days_12m < 200):
            continue
        target_dict = get_all_targets(price_data, curr_start_date, num_days_12m, 'target')
        sp500_dict = get_all_targets(price_data_sp500, curr_start_date, num_days_12m, 'sp500')
        target_dict.update(sp500_dict)
        target_df = pd.DataFrame.from_dict(target_dict, orient='index').T
        target_df['report_date'] = curr_report_date
        target_df['start_date'] = curr_start_date
        target_df['end_date'] = curr_end_date_12m
        df = pd.concat([df, target_df], ignore_index=True)
    df['symbol'] = symbol
    return df

def make_targets_all_symbols(start_date, end_date, config_dict):
    '''
    Function to return the complete dataframe for all symbols and all annual report date periods

    目的：為所有股票符號生成一個綜合的目標數據框。
    流程：
    - 讀取配置文件中的股票符號列表。
    - 對每個股票符號調用 make_targets 函數。
    - 將所有股票的目標數據合併到一個大的數據框中。
    '''
    symbol_names = [os.path.basename(folder) for folder in glob.glob(os.path.join(config_dict['annual_reports_pdf_save_directory'], '*')) \
                            if os.path.isdir(folder)]
    price_data_sp500 = openbb.stocks.load('^GSPC', start_date=start_date, end_date=end_date, 
                                        verbose=False)
    full_df = pd.DataFrame()
    #Iterate over all symbols in the directory
    for i, symbol in enumerate(symbol_names):
        df = make_targets(symbol, start_date, end_date, price_data_sp500, config_dict)
        full_df = pd.concat([full_df, df], ignore_index=True)
        print('Completed: {}/{}'.format(i+1, len(symbol_names)))
    return full_df

def get_normalized_column(df, col):
    '''
    Function to rank and then normalise a column in the df.
    Returns:
        Pandas DF with additional normalised column
    
    目的：將數據框中的列標準化。
    流程：
    - 使用 Pandas 的 rank 方法對指定列進行排名，然後進行標準化。
    - 使用正態分布的累積分佈函數（stats.norm.ppf）進一步轉換排名數據，使其分佈更接近標準正態分佈。
    '''
    new_col = col + '_normalised'
    preds = df.loc[:, col]
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)
    df.loc[:, new_col] = gauss_ranked_preds
    return df

def bin_targets(df, input_col, output_col, percentile_list, label_list):
    '''
    Function for binning target columns according to percentiles
    Args:
        input_col: target column to normalise
        output_col: Name of new normalised column
        percentile_list: Percentiles for binning
        label_list: labels aka bins
    Returns:
        Pandas DF with binned targets. Used for final ML model building

    目的：根據百分位數對目標列進行分箱並標籤化。
    流程：
    - 使用 Pandas 的 qcut 函數根據百分位數列表對數據進行分箱。
    - 將這些箱子映射到指定的標籤上，用於後續的機器學習模型訓練。
    '''
    s = df.loc[:, input_col]
    binned_series = pd.qcut(s, q=percentile_list, labels=label_list)
    label_list_float = [np.float32(x) for x in label_list]
    binned_series.replace(to_replace=label_list, value=label_list_float, inplace=True)
    df.loc[:, output_col] = binned_series.astype('float32')
    return df

def main(args):
    '''
    流程：
    - 加載配置文件並設定日期範圍。
    - 調用 make_targets_all_symbols 為所有股票生成目標數據。
    - 對數據進行清洗和標準化處理，然後將其保存為一個可供機器學習模型使用的格式。
    '''
    with open(args.config_path) as json_file:
        config_dict = json.load(json_file)
    start_date='2002-01-01'
    end_date='2023-12-31'
    targets_df = make_targets_all_symbols(start_date, end_date, config_dict)
    targets_df_filtered = targets_df.loc[lambda x: ~(x.isnull().any(axis=1))]
    #Create a column called era which denotes the year of annual report filing
    targets_df_filtered['era'] = targets_df_filtered['report_date'].apply(lambda x: x.year)
    #Drop duplicates if they exist. Could be if consecutive annual reports are published in same year.
    targets_df_filtered_dedup = targets_df_filtered.drop_duplicates(subset=['era', 'symbol']).reset_index(drop=True)
    target_cols = [c for c in targets_df_filtered_dedup.columns if c.startswith('target')]
    #Generate normalised target columns
    for target in target_cols:
        targets_df_filtered_dedup = targets_df_filtered_dedup.groupby('era', group_keys=False).apply(lambda df: \
                                                                        get_normalized_column(df, target))
    target_cols_normalised = [c for c in targets_df_filtered_dedup.columns if \
                                (c.startswith('target') & (c.endswith('normalised')))]
    #Create final target column for Machine Learning model building
    input_col_target = 'target_12m_normalised'
    output_col_target = 'target_ml'
    targets_df_filtered_dedup = targets_df_filtered_dedup.groupby('era', group_keys=False)\
                                    .apply(lambda df: bin_targets(df, input_col_target, output_col_target, 
                                                                    [0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                                                                    ['0.0', '0.25', '0.5', '0.75', '1.0']))
    with open(config_dict['targets_df_path'], 'wb') as handle:
        pickle.dump(targets_df_filtered_dedup, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', type=str,
                        required=True,
                        help='''Full path of config.json''')
    main(args=parser.parse_args())
    sys.exit(0)
