# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:12:14 2020

@author: xiajie jia
"""
from datetime import timedelta, date
import pandas as pd
from typing import List

__all__ = [
    '',
]

def kdata_resample(minutely_df: pd.DataFrame, k_freq:int = 60):
        
    df = minutely_df.copy()
    df['resample_time'] = df.datetime.apply(lambda r: (r - timedelta(minutes = 30)) if r.hour < 12 else (r - timedelta(minutes = 0)))
    df['resample_time'] = df.resample_time.apply(lambda r: (r - timedelta(seconds = 30)) if (r.hour in [11,14,15] and r.minute == 0) else r)
    df['resample_hour'] = df.apply(lambda r: timedelta(hours = r.resample_time.hour) + pd.to_datetime(r.date), axis = 1)
    df.index = df.resample_time
        
    open_adj_kfreq = pd.Series(df.open_adj.resample('{}min'.format(k_freq)).first(), name = 'open_adj_kfreq')
    high_adj_kfreq = pd.Series(df.high_adj.resample('{}min'.format(k_freq)).max(), name = 'high_adj_kfreq')
    low_adj_kfreq = pd.Series(df.low_adj.resample('{}min'.format(k_freq)).min(), name = 'low_adj_kfreq')
    close_adj_kfreq = pd.Series(df.close_adj.resample('{}min'.format(k_freq)).last(), name = 'close_adj_kfreq')
    amount_kfreq = pd.Series(df.amount.resample('{}min'.format(k_freq)).sum(), name = 'amount_kfreq')
    volume_kfreq = pd.Series(df.volume.resample('{}min'.format(k_freq)).sum(), name = 'volume_kfreq')
    pctchange_kfreq = pd.Series(df.pctchange.resample('{}min'.format(k_freq)).sum(), name = 'pctchange_kfreq')
    df_kfreq = pd.DataFrame([open_adj_kfreq,high_adj_kfreq,low_adj_kfreq,close_adj_kfreq,amount_kfreq,volume_kfreq,pctchange_kfreq]).T   
    df.index = df.resample_hour
    df = df.merge(df_kfreq, how = 'inner', left_index = True, right_index = True)
    df.reset_index(drop = True, inplace = True)
    return df

def prepare_basic_factor(daily_df:pd.DataFrame, periods:List[int]=[5,10,20,55,110])->pd.DataFrame:
    df = daily_df.copy()
    df.loc[:, 'swing'] = (df.high_adj - df.low_adj) / df.close_adj.shift()
    df.loc[:, 'vwap_ret'] = df.vwap_adj.pct_change()
    for T in periods:
        df.loc[:, f'mean_ret_{T}'] = df.pct_chg.rolling(window = T).mean()
        df.loc[:, f'mean_turn_{T}'] = df.turn.rolling(window = T).mean()
        df.loc[:, f'ret_{T}'] = df.close_adj.pct_change(T)
        df.loc[:, f'vol_ret_{T}'] = df.pct_chg.rolling(window = T).std()
        df.loc[:, f'kurt_ret_{T}'] = df.pct_chg.rolling(window = T).kurt()
        df.loc[:, f'swing_{T}'] = (df.high_adj.rolling(window = T).max() - df.low_adj.rolling(window = T).min()) / df.close_adj.shift(T)
        df.loc[:, f'mean_swing_{T}'] = df.swing.rolling(window = T).mean()
        df.loc[:, f'vol_swing_{T}'] = df.swing.rolling(window = T).std()
        df.loc[:, f'vwap_ret_{T}'] = df.vwap_adj.pct_change(T)
        df.loc[:, f'mean_vwap_ret_{T}'] = df.vwap_ret.rolling(window = T).mean()
        df.loc[:, f'vol_vwap_ret_{T}'] = df.vwap_ret.rolling(window = T).std()
    return df

def prepare_next_rets(daily_df:pd.DataFrame,periods:List[int]=[1,2,5,10,20,60])->pd.DataFrame:
    df = daily_df.copy()
    for T in periods:
        df.loc[:, f'next_{T}d_ret'] = df.close_adj.shift(-T) / df.close_adj - 1
    return df


def process_minutely_df(minutely_df:pd.DataFrame)->pd.DataFrame:
    minutely_df = minutely_df.copy()
    minutely_df.sort_values(by=['code','datetime'],inplace=True)
    minutely_df.loc[:,'date'] = minutely_df.datetime.apply(lambda r: date(r.year, r.month, r.day))
    minutely_df = minutely_df[(minutely_df.volume != 0) & (minutely_df.amount != 0)]
    minutely_df.reset_index(inplace=True,drop=True)
    return minutely_df

def process_daily_df(daily_df:pd.DataFrame)->pd.DataFrame:
    daily_df = daily_df.copy()
    daily_df.sort_values(by=['code','datetime'],inplace=True)
    daily_df.loc[:,'date'] = daily_df.datetime.apply(lambda r: date(r.year, r.month, r.day))
    daily_df.reset_index(inplace=True,drop=True)
    return daily_df