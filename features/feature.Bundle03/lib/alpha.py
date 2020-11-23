from typing import *
import numpy as np
import pandas as pd
import talib as ta

from ._base import *


# %%
# 超买超卖指标

class Tech_ATR_Factor(Factor):
    """真实波幅指标"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ATR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ATR_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_ATR.loc[:, f'Tech_ATR_{T}'] = ta.ATR(df.high_adj, df.low_adj, df.close_adj, timeperiod=T)

        return Tech_ATR


class Tech_BIAS_Factor(Factor):
    """乖离率"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_BIAS = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_BIAS_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_BIAS.loc[:, f'Tech_BIAS_{T}'] = (df.close_adj - df.close_adj.rolling(window=T).mean()) / df.close_adj.rolling(window=T).mean()

        return Tech_BIAS


class Tech_BOLL_Factor(Factor):
    """布林带"""
    T: int = 20
    num: int = 2
    matype: int = 0

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_BOLL = pd.DataFrame(np.nan, index=df['datetime'], columns=['Tech_BOLLUP', 'Tech_BOLLDOWN'])
        BOLL = ta.BBANDS(df.close_adj, timeperiod=self.T, nbdevup=self.num, nbdevdn=self.num, matype=self.matype)
        Tech_BOLL.loc[:, 'Tech_BOLLUP'] = BOLL[0]
        Tech_BOLL.loc[:, 'Tech_BOLLDOWN'] = BOLL[1]

        return Tech_BOLL


class Tech_SKEW_Factor(Factor):
    """收益偏度"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_SKEW = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_SKEW_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_SKEW.loc[:, f'Tech_SKEW_{T}'] = df.pct_chg.rolling(window=T).skew()

        return Tech_SKEW


class Tech_CCI_Factor(Factor):
    """顺势指标"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_CCI = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_CCI_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_CCI.loc[:, f'Tech_CCI_{T}'] = ta.CCI(df.high_adj, df.low_adj, df.close_adj, timeperiod=T)

        return Tech_CCI


class Tech_MFI_Factor(Factor):
    """现金流量指标"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_MFI = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_MFI_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_MFI.loc[:, f'Tech_MFI_{T}'] = ta.MFI(df.high_adj, df.low_adj, df.close_adj, df.volume, timeperiod=T)

        return Tech_MFI


class Tech_ROC_Factor(Factor):
    """变动速率指标"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ROC = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ROC_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_ROC.loc[:, f'Tech_ROC_{T}'] = ta.ROC(df.close_adj, timeperiod=T)
        return Tech_ROC


class Tech_RSI_Factor(Factor):
    """相对强弱指标"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_RSI = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_RSI_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_RSI.loc[:, f'Tech_RSI_{T}'] = ta.RSI(df.close_adj, timeperiod=T)

        return Tech_RSI


class Tech_WILLR_Factor(Factor):
    """威廉指标"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_WILLR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_WILLR_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_WILLR.loc[:, f'Tech_WILLR_{T}'] = ta.WILLR(df.high_adj, df.low_adj, df.close_adj, timeperiod=T)

        return Tech_WILLR


class Tech_CMO_Factor(Factor):
    """钱德动量摆动指标"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_CMO = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_CMO_{self.T_list[0]}'])
        df.loc[:, 'delta'] = df.close_adj - df.close_adj.shift()
        df.loc[:, 'SU'] = df.delta.apply(lambda r: r if r > 0 else 0)
        df.loc[:, 'SD'] = df.delta.apply(lambda r: -r if r < 0 else 0)
        for T in self.T_list:
            Tech_CMO.loc[:, f'Tech_CMO_{T}'] = (
                (df.SU.rolling(window=T).sum() - df.SD.rolling(window=T).sum()) /
                (df.SU.rolling(window=T).sum() + df.SD.rolling(window=T).sum())
            )

        return Tech_CMO


# %%
# 趋势指标
class Tech_AD_Factor(Factor):
    """派发指标及其均线"""
    T_list: List[int] = [5, 10,20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_AD = pd.DataFrame(np.nan, index=df['datetime'], columns=['Tech_AD'])
        Tech_AD.loc[:, 'Tech_AD'] = ta.AD(df.high_adj, df.low_adj, df.close_adj, df.volume)
        for T in self.T_list:
            Tech_AD.loc[:, f'Tech_AD_{T}']= Tech_AD.rolling(window=T).mean()
        return Tech_AD


class Tech_ADX_Factor(Factor):
    """平均动向指数"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ADX = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ADX_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_ADX.loc[:, f'Tech_ADX_{T}'] = ta.ADX(df.high_adj, df.low_adj, df.close_adj, timeperiod=T)
        return Tech_ADX


class Tech_ADXR_Factor(Factor):
    """相对平均动向指数"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ADXR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ADXR_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_ADXR.loc[:, f'Tech_ADXR_{T}'] = ta.ADXR(df.high_adj, df.low_adj, df.close_adj, timeperiod=T)
        return Tech_ADXR


class Tech_AROON_Factor(Factor):
    """阿隆指数, 返回aroon up 与aroon down"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_AROON = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_AROONUP_{self.T_list[0]}', f'Tech_AROONDOWN_{self.T_list[0]}'])
        for T in self.T_list:
            result = ta.AROON(df.high_adj, df.low_adj, timeperiod=T)
            Tech_AROON.loc[:, f'Tech_AROONUP_{T}'] = result[0]
            Tech_AROON.loc[:, f'Tech_AROONDOWN_{T}'] = result[1]
        return Tech_AROON


class Tech_MDV_Factor(Factor):
    """波幅中位数"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_MDV = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_MDV_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_MDV.loc[:, f'Tech_MDV_{T}'] = np.log(df.high_adj / df.low_adj).rolling(window=T).median()
        return Tech_MDV


class Tech_ULCER_Factor(Factor):
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ULCER = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ULCER_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_ULCER.loc[:, f'Tech_ULCER_{T}'] = (df.close_adj - df.close_adj.rolling(window=T).max()).rolling(window=T).std()
        return Tech_ULCER


class Tech_MTM_Factor(Factor):
    """动量指标, 近似mom"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_MTM = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_MTM_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_MTM.loc[:, f'Tech_MTM_{T}'] = ta.MOM(df.close_adj, timeperiod=T)
        return Tech_MTM


class Tech_DDI_Factor(Factor):
    """方向标准差离差指数"""
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_DDI = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_DDI_{self.T_list[0]}'])
        df.loc[:, 'addhl'] = df.high_adj + df.low_adj
        df.loc[:, 'pre_addhl'] = df.addhl.shift()
        df.loc[:, 'pre_high_adj'] = df.high_adj.shift()
        df.loc[:, 'pre_low_adj'] = df.low_adj.shift()
        df.loc[:, 'dmzi'] = df.apply(lambda r: max(abs(r.high_adj - r.pre_high_adj), abs(r.low_adj - r.pre_low_adj)),axis=1)
        df.loc[:, 'dmz'] = df.apply(lambda r: r.dmzi if r.addhl > r.pre_addhl else 0, axis=1)
        df.loc[:, 'dmf'] = df.apply(lambda r: 0 if r.addhl >= r.pre_addhl else r.dmzi, axis=1)
        for T in self.T_list:
            Tech_DDI.loc[:, f'Tech_DDI_{T}'] = (df.dmz.rolling(window=T).sum() - df.dmf.rolling(window=T).sum()) / (
                        df.dmz.rolling(window=T).sum() + df.dmf.rolling(window=T).sum())

        return Tech_DDI


class Tech_CHO_Factor(Factor):
    """佳庆指标"""
    T_fast_list: List[int] = [3, 6, 9, 12]
    T_slow_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_CHO = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_CHO_{self.T_fast_list[0]}_{self.T_slow_list[0]}'])
        for T in zip(self.T_fast_list, self.T_slow_list):
            Tech_CHO.loc[:, f'Tech_CHO_{T[0]}_{T[1]}'] = ta.ADOSC(df.high_adj, df.low_adj, df.close_adj, df.volume, fastperiod=T[0], slowperiod=T[1])

        return Tech_CHO


class Tech_TRIX_Factor(Factor):
    """"""
    T_list: List[int] = [10, 20, 30, 55, 110]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_TRIX = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_TRIX_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_TRIX.loc[:, f'Tech_TRIX_{T}'] = ta.TRIX(df.close_adj, timeperiod=T)

        return Tech_TRIX


# %%
# 能量指标
class Tech_ELDER_Factor(Factor):
    T_list: List[int] = [7, 14, 28, 56]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ELDER = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ELDER_{self.T_list[0]}'])
        for T in self.T_list:
            df.loc[:, 'high_ema'] = ta.EMA(df.high_adj, timeperiod=T)
            df.loc[:, 'low_ema'] = ta.EMA(df.low_adj, timeperiod=T)
            Tech_ELDER.loc[:, f'Tech_ELDER_{T}'] = ((df.high_adj - df.high_ema) - (df.low_adj - df.low_ema)) / df.close_adj

        return Tech_ELDER


class Tech_MASS_Factor(Factor):
    T_fast_list: List[int] = [6, 9, 18, 36]
    T_slow_list: List[int] = [15, 25, 50, 100]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_MASS = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_MASS_{self.T_fast_list[0]}_{self.T_slow_list[0]}'])
        for T in zip(self.T_fast_list, self.T_slow_list):
            df.loc[:, 'AHL'] = (df.high_adj - df.low_adj).rolling(window=T[0]).mean()
            df.loc[:, 'BHL'] = (df.AHL).rolling(window=T[0]).mean()
            Tech_MASS.loc[:, f'Tech_MASS_{T[0]}_{T[1]}'] = (df.AHL / df.BHL).rolling(window=T[1]).sum()

        return Tech_MASS


class Tech_CR_Factor(Factor):
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_CR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_CR_{self.T_list[0]}'])
        df.loc[:, 'PM'] = (df.high_adj + df.low_adj + df.close_adj) / 3
        for T in self.T_list:
            Tech_CR.loc[:, f'Tech_CR_{T}'] = (df.high_adj - df.PM).rolling(window=T).sum() / (df.PM - df.low_adj).rolling(window=T).sum()
        return Tech_CR


class Tech_BR_Factor(Factor):
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_BR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_BR_{self.T_list[0]}'])
        df.loc[:, 'pre_close_adj'] = df.close_adj.shift()
        for T in self.T_list:
            Tech_BR.loc[:, f'Tech_BR_{T}'] = (df.high_adj - df.pre_close_adj).rolling(window=T).sum() / (
                                              df.pre_close_adj - df.low_adj).rolling(window=T).sum()
        return Tech_BR


class Tech_AR_Factor(Factor):
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_AR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_AR_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_AR.loc[:, f'Tech_AR_{T}'] = (df.high_adj - df.open_adj).rolling(window=T).sum() / (
                        df.open_adj - df.low_adj).rolling(window=T).sum()
        return Tech_AR


# %%
# 成交量指标
class Tech_OBV_Factor(Factor):
    """能量潮"""
    T_list: List[int] = [5, 10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_OBV = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_OBV_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_OBV.loc[:, 'Tech_OBV'] = ta.OBV(df.close_adj, df.volume).rolling(window=T).mean()

        return Tech_OBV


class Tech_RETURN_Factor(Factor):
    """相对换手率指标"""
    T_fast_list: List[int] = [10, 20, 30, 55]
    T_slow_list: List[int] = [60, 120, 180, 240]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_RETURN = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_RETURN_{self.T_fast_list[0]}_{self.T_slow_list[0]}'])
        for T in zip(self.T_fast_list, self.T_slow_list):
            Tech_RETURN.loc[:, f'Tech_RETURN_{T[0]}_{T[1]}'] = df.turn.rolling(window=T[0]).mean() / df.turn.rolling(window=T[1]).mean()

        return Tech_RETURN


class Tech_VEMA_Factor(Factor):
    """成交量EMA"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_VEMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_VEMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_VEMA.loc[:, f'Tech_VEMA_{T}'] = ta.EMA(df.volume, timeperiod=T)

        return Tech_VEMA


class Tech_AEMA_Factor(Factor):
    """成交额EMA"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_AEMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_AEMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_AEMA.loc[:, f'Tech_AEMA_{T}'] = ta.EMA(df.amount, timeperiod=T)

        return Tech_AEMA


class Tech_VSMA_Factor(Factor):
    """成交量SMA"""
    T_list: List[int] = [10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_VSMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_VSMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_VSMA.loc[:, f'Tech_VSMA_{T}'] = df.volume.rolling(window=T).mean()

        return Tech_VSMA


class Tech_ASMA_Factor(Factor):
    """成交额SMA"""
    T_list: List[int] = [5, 10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ASMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_ASMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_ASMA.loc[:, f'Tech_ASMA_{T}'] = df.volume.rolling(window=T).mean()

        return Tech_ASMA


class Tech_STOM_Factor(Factor):
    """月换手率对数"""
    T_list: List[int] = [5, 10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_STOM = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_STOM_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_STOM.loc[:, f'Tech_STOM_{T}'] = np.log(df.turn.rolling(window=T).sum())

        return Tech_STOM


class Tech_VR_Factor(Factor):
    T_list: List[int] = [5, 10, 20, 30, 55]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_VR = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_VR_{self.T_list[0]}'])

        def func(y, df):
            x = df.loc[y.index, 'pct_chg']
            up = y.loc[x[x > 0].index].mean()
            down = y.loc[x[x < 0].index].mean()
            return up / down
        for T in self.T_list:
            Tech_VR.loc[:, f'Tech_VR_{T}'] = df.volume.rolling(window=T).apply(lambda r: func(r, df), raw=False)

        return Tech_VR


class Tech_DISTANCE_Factor(Factor):
    """与过去N日最高收盘价的距离"""
    T_list: List[int] = [40, 80, 120, 160, 200]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_DISTANCE = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_DISTANCE_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_DISTANCE.loc[:, f'Tech_DISTANCE'] = 1 - df.close_adj / df.close_adj.rolling(window=T).max()

        return Tech_DISTANCE


class Tech_MOMAVG_Factor(Factor):
    """过去1、3、6、12月的平均涨幅"""

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_MOMAVG = pd.DataFrame(np.nan, index=df['datetime'], columns=['Tech_MOMAVG'])
        Tech_MOMAVG.loc[:, 'Tech_MOMAVG'] = 0.25 * (df.close_adj.pct_change(20) + df.close_adj.pct_change(60) +
                                                    df.close_adj.pct_change(120) + df.close_adj.pct_change(240))

        return Tech_MOMAVG


class Tech_ULTOSC_Factor(Factor):
    """终极振子"""

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_ULTOSC = pd.DataFrame(np.nan, index=df['datetime'], columns=['Tech_ULTOSC'])
        Tech_ULTOSC.loc[:, 'Tech_ULTOSC'] = ta.ULTOSC(df.high_adj, df.low_adj, df.close_adj,
                                                      timeperiod1=7, timeperiod2=14, timeperiod3=28)
        return Tech_ULTOSC


# %%
# 均线指标
class Tech_SMA_Factor(Factor):
    """默认简单移动平均线"""
    T_list: List[int] = [5, 10, 20, 30, 55, 110]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_SMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_SMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_SMA.loc[:, f'Tech_SMA_{T}'] = ta.MA(df.close_adj, timeperiod=T, matype=0)

        return Tech_SMA


class Tech_EMA_Factor(Factor):
    """指数加权平均线"""
    T_list: List[int] = [5, 10, 20, 30, 55, 110]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_EMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_EMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_EMA.loc[:, f'Tech_EMA_{T}'] = ta.EMA(df.close_adj, timeperiod=T)

        return Tech_EMA


class Tech_KAMA_Factor(Factor):
    """自适应均线"""
    T_list: List[int] = [5, 10, 20, 30, 55, 110]

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Tech_KAMA = pd.DataFrame(np.nan, index=df['datetime'], columns=[f'Tech_KAMA_{self.T_list[0]}'])
        for T in self.T_list:
            Tech_KAMA.loc[:, f'Tech_KAMA_{T}'] = ta.KAMA(df.close_adj, timeperiod=T)

        return Tech_KAMA


# 形态指标
class Patn_2CROWS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_2CROWS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_2CROWS'])
        Patn_2CROWS.loc[:, 'Patn_2CROWS'] = ta.CDL2CROWS(df.open_adj, df.high_adj, df.low_adj, df.close_adj)

        return Patn_2CROWS


class Patn_3BLACKCROWS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_3BLACKCROWS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_3BLACKCROWS'])
        Patn_3BLACKCROWS.loc[:, 'Patn_3BLACKCROWS'] = ta.CDL3BLACKCROWS(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_3BLACKCROWS


class Patn_3INSIDE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_3INSIDE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_3INSIDE']) / 100

        return Patn_3INSIDE


class Patn_3LINESTRIKE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_3LINESTRIKE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_3LINESTRIKE'])
        Patn_3LINESTRIKE.loc[:, 'Patn_3LINESTRIKE'] = ta.CDL3LINESTRIKE(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_3LINESTRIKE


class Patn_3OUTSIDE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_3OUTSIDE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_3OUTSIDE'])
        Patn_3OUTSIDE.loc[:, 'Patn_3OUTSIDE'] = ta.CDL3OUTSIDE(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_3OUTSIDE


class Patn_3STARSINSOUTH_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_3STARSINSOUTH = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_3STARSINSOUTH'])
        Patn_3STARSINSOUTH.loc[:, 'Patn_3STARSINSOUTH'] = ta.CDL3STARSINSOUTH(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_3STARSINSOUTH


class Patn_3WHITESOLDIERS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_3WHITESOLDIERS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_3WHITESOLDIERS'])
        Patn_3WHITESOLDIERS.loc[:, 'Patn_3WHITESOLDIERS'] = ta.CDL3WHITESOLDIERS(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_3WHITESOLDIERS


class Patn_ABANDONEDBABY_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_ABANDONEDBABY = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_ABANDONEDBABY'])
        Patn_ABANDONEDBABY.loc[:, 'Patn_ABANDONEDBABY'] = ta.CDLABANDONEDBABY(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_ABANDONEDBABY


class Patn_ADVANCEBLOCK_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_ADVANCEBLOCK = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_ADVANCEBLOCK'])
        Patn_ADVANCEBLOCK.loc[:, 'Patn_ADVANCEBLOCK'] = ta.CDLADVANCEBLOCK(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_ADVANCEBLOCK


class Patn_BELTHOLD_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_BELTHOLD = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_BELTHOLD'])
        Patn_BELTHOLD.loc[:, 'Patn_BELTHOLD'] = ta.CDLBELTHOLD(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_BELTHOLD


class Patn_BREAKAWAY_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_BREAKAWAY = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_BREAKAWAY'])
        Patn_BREAKAWAY.loc[:, 'Patn_BREAKAWAY'] = ta.CDLBREAKAWAY(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_BREAKAWAY


class Patn_CLOSINGMARUBOZU_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_CLOSINGMARUBOZU = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_CLOSINGMARUBOZU'])
        Patn_CLOSINGMARUBOZU.loc[:, 'Patn_CLOSINGMARUBOZU'] = ta.CDLCLOSINGMARUBOZU(df.open_adj, df.high_adj,
                                                                                    df.low_adj, df.close_adj) / 100

        return Patn_CLOSINGMARUBOZU


class Patn_CONCEALBABYSWALL_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_CONCEALBABYSWALL = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_CONCEALBABYSWALL'])
        Patn_CONCEALBABYSWALL.loc[:, 'Patn_CONCEALBABYSWALL'] = ta.CDLCONCEALBABYSWALL(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_CONCEALBABYSWALL


class Patn_COUNTERATTACK_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_COUNTERATTACK = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_COUNTERATTACK'])
        Patn_COUNTERATTACK.loc[:, 'Patn_COUNTERATTACK'] = ta.CDLCOUNTERATTACK(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_COUNTERATTACK


class Patn_DARKCLOUDCOVER_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_DARKCLOUDCOVER = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_DARKCLOUDCOVER'])
        Patn_DARKCLOUDCOVER.loc[:, 'Patn_DARKCLOUDCOVER'] = ta.CDLDARKCLOUDCOVER(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_DARKCLOUDCOVER


class Patn_DOJI_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_DOJI = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_DOJI'])
        Patn_DOJI.loc[:, 'Patn_DOJI'] = ta.CDLDOJI(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_DOJI


class Patn_DOJISTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_DOJISTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_DOJISTAR'])
        Patn_DOJISTAR.loc[:, 'Patn_DOJISTAR'] = ta.CDLDOJISTAR(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_DOJISTAR


class Patn_DRAGONFLYDOJI_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_DRAGONFLYDOJI = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_DRAGONFLYDOJI'])
        Patn_DRAGONFLYDOJI.loc[:, 'Patn_DRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_DRAGONFLYDOJI


class Patn_ENGULFING_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_ENGULFING = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_ENGULFING'])
        Patn_ENGULFING.loc[:, 'Patn_ENGULFING'] = ta.CDLENGULFING(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_ENGULFING


class Patn_EVENINGDOJISTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_EVENINGDOJISTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_EVENINGDOJISTAR'])
        Patn_EVENINGDOJISTAR.loc[:, 'Patn_EVENINGDOJISTAR'] = ta.CDLEVENINGDOJISTAR(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_EVENINGDOJISTAR


class Patn_EVENINGSTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_EVENINGSTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_EVENINGSTAR'])
        Patn_EVENINGSTAR.loc[:, 'Patn_EVENINGSTAR'] = ta.CDLEVENINGSTAR(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_EVENINGSTAR


class Patn_GAPSIDESIDEWHITE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_GAPSIDESIDEWHITE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_GAPSIDESIDEWHITE'])
        Patn_GAPSIDESIDEWHITE.loc[:, 'Patn_GAPSIDESIDEWHITE'] = ta.CDLGAPSIDESIDEWHITE(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_GAPSIDESIDEWHITE


class Patn_GRAVESTONEDOJI_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_GRAVESTONEDOJI = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_GRAVESTONEDOJI'])
        Patn_GRAVESTONEDOJI.loc[:, 'Patn_GRAVESTONEDOJI'] = ta.CDLGRAVESTONEDOJI(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_GRAVESTONEDOJI


class Patn_HAMMER_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HAMMER = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HAMMER'])
        Patn_HAMMER.loc[:, 'Patn_HAMMER'] = ta.CDLHAMMER(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HAMMER


class Patn_HANGINGMAN_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HANGINGMAN = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HANGINGMAN'])
        Patn_HANGINGMAN.loc[:, 'Patn_HANGINGMAN'] = ta.CDLHANGINGMAN(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HANGINGMAN


class Patn_HARAMI_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HARAMI = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HARAMI'])
        Patn_HARAMI.loc[:, 'Patn_HARAMI'] = ta.CDLHARAMI(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HARAMI


class Patn_HARAMICROSS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HARAMICROSS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HARAMICROSS'])
        Patn_HARAMICROSS.loc[:, 'Patn_HARAMICROSS'] = ta.CDLHARAMICROSS(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HARAMICROSS


class Patn_HIGHWAVE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HIGHWAVE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HIGHWAVE'])
        Patn_HIGHWAVE.loc[:, 'Patn_HIGHWAVE'] = ta.CDLHIGHWAVE(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HIGHWAVE


class Patn_HIKKAKE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HIKKAKE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HIKKAKE'])
        Patn_HIKKAKE.loc[:, 'Patn_HIKKAKE'] = ta.CDLHIKKAKE(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HIKKAKE


class Patn_HIKKAKEMOD_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HIKKAKEMOD = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HIKKAKEMOD'])
        Patn_HIKKAKEMOD.loc[:, 'Patn_HIKKAKEMOD'] = ta.CDLHIKKAKEMOD(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HIKKAKEMOD


class Patn_HOMINGPIGEON_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_HOMINGPIGEON = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_HOMINGPIGEON'])
        Patn_HOMINGPIGEON.loc[:, 'Patn_HOMINGPIGEON'] = ta.CDLHOMINGPIGEON(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_HOMINGPIGEON


class Patn_IDENTICAL3CROWS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_IDENTICAL3CROWS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_IDENTICAL3CROWS'])
        Patn_IDENTICAL3CROWS.loc[:, 'Patn_IDENTICAL3CROWS'] = ta.CDLIDENTICAL3CROWS(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_IDENTICAL3CROWS


class Patn_INNECK_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_INNECK = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_INNECK'])
        Patn_INNECK.loc[:, 'Patn_INNECK'] = ta.CDLINNECK(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_INNECK


class Patn_INVERTEDHAMMER_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_INVERTEDHAMMER = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_INVERTEDHAMMER'])
        Patn_INVERTEDHAMMER.loc[:, 'Patn_INVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_INVERTEDHAMMER


class Patn_KICKING_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_KICKING = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_KICKING'])
        Patn_KICKING.loc[:, 'Patn_KICKING'] = ta.CDLKICKING(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_KICKING


class Patn_KICKINGBYLENGTH_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_KICKINGBYLENGTH = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_KICKINGBYLENGTH'])
        Patn_KICKINGBYLENGTH.loc[:, 'Patn_KICKINGBYLENGTH'] = ta.CDLKICKINGBYLENGTH(df.open_adj, df.high_adj,
                                                                                    df.low_adj, df.close_adj) / 100

        return Patn_KICKINGBYLENGTH


class Patn_LADDERBOTTOM_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_LADDERBOTTOM = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_LADDERBOTTOM'])
        Patn_LADDERBOTTOM.loc[:, 'Patn_LADDERBOTTOM'] = ta.CDLLADDERBOTTOM(df.open_adj, df.high_adj, df.low_adj,
                                                                           df.close_adj) / 100

        return Patn_LADDERBOTTOM


class Patn_LONGLEGGEDDOJI_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_LONGLEGGEDDOJI = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_LONGLEGGEDDOJI'])
        Patn_LONGLEGGEDDOJI.loc[:, 'Patn_LONGLEGGEDDOJI'] = ta.CDLLONGLEGGEDDOJI(df.open_adj, df.high_adj, df.low_adj,
                                                                                 df.close_adj) / 100

        return Patn_LONGLEGGEDDOJI


class Patn_LONGLINE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_LONGLINE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_LONGLINE'])
        Patn_LONGLINE.loc[:, 'Patn_LONGLINE'] = ta.CDLLONGLINE(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_LONGLINE


class Patn_MARUBOZU_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_MARUBOZU = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_MARUBOZU'])
        Patn_MARUBOZU.loc[:, 'Patn_MARUBOZU'] = ta.CDLMARUBOZU(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_MARUBOZU


class Patn_MATCHINGLOW_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_MATCHINGLOW = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_MATCHINGLOW'])
        Patn_MATCHINGLOW.loc[:, 'Patn_MATCHINGLOW'] = ta.CDLMATCHINGLOW(df.open_adj, df.high_adj, df.low_adj,
                                                                        df.close_adj) / 100

        return Patn_MATCHINGLOW


class Patn_MATHOLD_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_MATHOLD = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_MATHOLD'])
        Patn_MATHOLD.loc[:, 'Patn_MATHOLD'] = ta.CDLMATHOLD(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_MATHOLD


class Patn_MORNINGDOJISTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_MORNINGDOJISTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_MORNINGDOJISTAR'])
        Patn_MORNINGDOJISTAR.loc[:, 'Patn_MORNINGDOJISTAR'] = ta.CDLMORNINGDOJISTAR(df.open_adj, df.high_adj,
                                                                                    df.low_adj, df.close_adj) / 100

        return Patn_MORNINGDOJISTAR


class Patn_MORNINGSTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_MORNINGSTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_MORNINGSTAR'])
        Patn_MORNINGSTAR.loc[:, 'Patn_MORNINGSTAR'] = ta.CDLMORNINGSTAR(df.open_adj, df.high_adj, df.low_adj,
                                                                        df.close_adj) / 100

        return Patn_MORNINGSTAR


class Patn_ONNECK_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_ONNECK = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_ONNECK'])
        Patn_ONNECK.loc[:, 'Patn_ONNECK'] = ta.CDLONNECK(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_ONNECK


class Patn_PIERCING_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_PIERCING = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_PIERCING'])
        Patn_PIERCING.loc[:, 'Patn_PIERCING'] = ta.CDLPIERCING(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_PIERCING


class Patn_RICKSHAWMAN_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_RICKSHAWMAN = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_RICKSHAWMAN'])
        Patn_RICKSHAWMAN.loc[:, 'Patn_RICKSHAWMAN'] = ta.CDLRICKSHAWMAN(df.open_adj, df.high_adj, df.low_adj,
                                                                        df.close_adj) / 100

        return Patn_RICKSHAWMAN


class Patn_RISEFALL3METHODS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_RISEFALL3METHODS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_RISEFALL3METHODS'])
        Patn_RISEFALL3METHODS.loc[:, 'Patn_RISEFALL3METHODS'] = ta.CDLRISEFALL3METHODS(df.open_adj, df.high_adj,
                                                                                       df.low_adj, df.close_adj) / 100

        return Patn_RISEFALL3METHODS


class Patn_SEPARATINGLINES_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_SEPARATINGLINES = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_SEPARATINGLINES'])
        Patn_SEPARATINGLINES.loc[:, 'Patn_SEPARATINGLINES'] = ta.CDLSEPARATINGLINES(df.open_adj, df.high_adj,
                                                                                    df.low_adj, df.close_adj) / 100

        return Patn_SEPARATINGLINES


class Patn_SHOOTINGSTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_SHOOTINGSTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_SHOOTINGSTAR'])
        Patn_SHOOTINGSTAR.loc[:, 'Patn_SHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df.open_adj, df.high_adj, df.low_adj,
                                                                           df.close_adj) / 100

        return Patn_SHOOTINGSTAR


class Patn_SHORTLINE_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_SHORTLINE = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_SHORTLINE'])
        Patn_SHORTLINE.loc[:, 'Patn_SHORTLINE'] = ta.CDLSHORTLINE(df.open_adj, df.high_adj, df.low_adj,
                                                                  df.close_adj) / 100

        return Patn_SHORTLINE


class Patn_SPINNINGTOP_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_SPINNINGTOP = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_SPINNINGTOP'])
        Patn_SPINNINGTOP.loc[:, 'Patn_SPINNINGTOP'] = ta.CDLSPINNINGTOP(df.open_adj, df.high_adj, df.low_adj,
                                                                        df.close_adj) / 100

        return Patn_SPINNINGTOP


class Patn_STALLEDPATTERN_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_STALLEDPATTERN = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_STALLEDPATTERN'])
        Patn_STALLEDPATTERN.loc[:, 'Patn_STALLEDPATTERN'] = ta.CDLSTALLEDPATTERN(df.open_adj, df.high_adj, df.low_adj,
                                                                                 df.close_adj) / 100

        return Patn_STALLEDPATTERN


class Patn_STICKSANDWICH_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_STICKSANDWICH = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_STICKSANDWICH'])
        Patn_STICKSANDWICH.loc[:, 'Patn_STICKSANDWICH'] = ta.CDLSTICKSANDWICH(df.open_adj, df.high_adj, df.low_adj,
                                                                              df.close_adj) / 100

        return Patn_STICKSANDWICH


class Patn_TAKURI_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_TAKURI = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_TAKURI'])
        Patn_TAKURI.loc[:, 'Patn_TAKURI'] = ta.CDLTAKURI(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_TAKURI


class Patn_TASUKIGAP_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_TASUKIGAP = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_TASUKIGAP'])
        Patn_TASUKIGAP.loc[:, 'Patn_TASUKIGAP'] = ta.CDLTASUKIGAP(df.open_adj, df.high_adj, df.low_adj,
                                                                  df.close_adj) / 100

        return Patn_TASUKIGAP


class Patn_THRUSTING_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_THRUSTING = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_THRUSTING'])
        Patn_THRUSTING.loc[:, 'Patn_THRUSTING'] = ta.CDLTHRUSTING(df.open_adj, df.high_adj, df.low_adj,
                                                                  df.close_adj) / 100

        return Patn_THRUSTING


class Patn_TRISTAR_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_TRISTAR = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_TRISTAR'])
        Patn_TRISTAR.loc[:, 'Patn_TRISTAR'] = ta.CDLTRISTAR(df.open_adj, df.high_adj, df.low_adj, df.close_adj) / 100

        return Patn_TRISTAR


class Patn_UNIQUE3RIVER_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_UNIQUE3RIVER = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_UNIQUE3RIVER'])
        Patn_UNIQUE3RIVER.loc[:, 'Patn_UNIQUE3RIVER'] = ta.CDLUNIQUE3RIVER(df.open_adj, df.high_adj, df.low_adj,
                                                                           df.close_adj) / 100

        return Patn_UNIQUE3RIVER


class Patn_UPSIDEGAP2CROWS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_UPSIDEGAP2CROWS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_UPSIDEGAP2CROWS'])
        Patn_UPSIDEGAP2CROWS.loc[:, 'Patn_UPSIDEGAP2CROWS'] = ta.CDLUPSIDEGAP2CROWS(df.open_adj, df.high_adj,
                                                                                    df.low_adj, df.close_adj) / 100

        return Patn_UPSIDEGAP2CROWS


class Patn_XSIDEGAP3METHODS_Factor(Factor):

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df.index = df['datetime']
        Patn_XSIDEGAP3METHODS = pd.DataFrame(np.nan, index=df['datetime'], columns=['Patn_XSIDEGAP3METHODS'])
        Patn_XSIDEGAP3METHODS.loc[:, 'Patn_XSIDEGAP3METHODS'] = ta.CDLXSIDEGAP3METHODS(df.open_adj, df.high_adj,
                                                                                       df.low_adj, df.close_adj) / 100

        return Patn_XSIDEGAP3METHODS
