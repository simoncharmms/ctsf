#### ==========================================================================
#### Dissertation utils
#### Author: Simon Schramm
#### 11.10.2021
#### --------------------------------------------------------------------------
""" 
This script stores utils for the dissertation.
""" 
### ---------------------------------------------------------------------------
### Preamble.
### ---------------------------------------------------------------------------
import os 
import time
import sys
import pandas as pd
#### --------------------------------------------------------------------------
#### Define function for reading csv-files with varying encoding.
### ---------------------------------------------------------------------------
def tik():
    # blockPrint()
    start_time = time.time()
    print('================================================================= \n' + 
        'Script \n' + sys.argv[0] + '\n' + 'started at: ' + 
        time.strftime('%H:%M:%S', time.gmtime(start_time)) + '. \n' + 
        '-----------------------------------------------------------------')
    # next_script = '... .py'
    return start_time
def tok(start_time):
    # enablePrint()
    elapsed_time = time.time() - start_time
    print('---------------------------------------------------------------- \n' + 
        'Elapsed time: ' + 
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time)) + 
        '. \n' + 
        '================================================================')
    # os.chdir(project_path)
    # os.system(next_script)
#### --------------------------------------------------------------------------
#### Define function for reading csv-files with varying encoding.
### ---------------------------------------------------------------------------
# Decoding issues with Zotero-csv-files requires to test different encodings.
def read_csv(fp):
     if os.path.splitext(fp)[1] != '.csv':
          return  # Could be extended to .xls, ...
     seps = [',', ';'] # ',' is default
     # ISO-8859-1 encoding is required for IHS-csv-files, could be extended.
     encodings = [None, 'utf-8', 'ISO-8859-1']
     for sep in seps:
         for encoding in encodings:
              try:
                  return pd.read_csv(fp, encoding=encoding, sep=sep, low_memory=False)
              except Exception:
                  pass
     raise ValueError("{!r}-property is not in encodings {} or seperators {}."
                      .format(fp, encodings, seps))
#### --------------------------------------------------------------------------
#### Define function to load data.
### ---------------------------------------------------------------------------
def get_data(FP_DATA):
    #  Get monthly automotive sales data
    df_monthly = pd.read_csv(FP_DATA+"data_chapter_2_monthly.csv", delimiter=",")
    # Convert the Timestamp column to datetime format
    df_monthly['Timestamp'] = pd.to_datetime(df_monthly['Timestamp'], format='%d.%m.%y')
    # Set the timestamp column as the index
    df_monthly = df_monthly.set_index('Timestamp')
    df_monthly.index.freq = pd.infer_freq(df_monthly.index)
    list_econ_cols = [
        'GDP in Germany Annually',
        'Average Income in Germany Annually', 
        'Expenditures for alcoholic beverages in Germany Annually', 
        'Personal Savings in Germany Quarterly',
        'Disposable Personal Income in Germany Quarterly',
        'Consumer Spending in Germany Quarterly',
        'Unemployment Rate in Germany Monthly',
        'Consumer Confidence in Germany Monthly',
        'Car Production in Germany Monthly',
        'Car Registrations in Germany Monthly',
        'Capacity Utilization in Germany Monthly', 
        'Business Confidence in Germany Monthly',
        'Bankruptcies in Germany Monthly'
        ]
    # Index economic columns to their earliest value
    for col in list_econ_cols:
        if col in df_monthly.columns:
            earliest_value = df_monthly[col].dropna().iloc[0]
            df_monthly[col + '_indexed'] = df_monthly[col] / earliest_value * 100
    # Get event data.
    df_events = pd.read_excel(FP_DATA + 'data_events.xlsx')
    return df_monthly, df_events
#
def get_data_filtered(df_monthly, start, end):
    df_monthly_filtered = df_monthly[[column]][(df_monthly.index >= pd.to_datetime(start)) & (df_monthly.index < pd.to_datetime(end))]
    return df_monthly
### ---------------------------------------------------------------------------
### End.
#### ==========================================================================