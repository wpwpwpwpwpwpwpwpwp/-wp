
import numpy as np
import pandas as pd
from scipy import stats
from .utils import kdata_resample,prepare_next_rets,prepare_basic_factor
from ._base import *


class BASIC_Info(Factor):

    factor_periods:list = [5, 10, 20, 30, 55]
    nextret_periods:list = [1, 2, 3, 5, 10, 20]

    def _call(self, code:str, daily_df: pd.DataFrame, minutely_df:pd.DataFrame):
        df = daily_df.copy()
        df = prepare_basic_factor(df, self.factor_periods)
        df = prepare_next_rets(df, self.nextret_periods)
        df.set_index(['date','code'],inplace=True)
        return df



class ARPFactor(Factor):

    T: int = 27
    back_step: int = 23

    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
        
        df = minutely_df.copy()
        ARP = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns=['ARP'])
        df['group_indexer'] = df.groupby(df.date).ngroup()
        num_of_group = df.group_indexer.max() + 1
        for k in range(self.T, num_of_group):
            temping = df.loc[(df.group_indexer > (k - self.T)) & (df.group_indexer <= k), :]
            rpp = temping.close_adj.sub(temping.low_adj.min()).div(temping.high_adj.max() - temping.low_adj.min())
            date_now = temping.loc[temping.group_indexer == k, 'date'].iat[0]
            ARP.at[date_now, 'ARP'] = rpp.sum()
        ARP.loc[:,'ARP'] = ARP.ARP.rolling(self.back_step).mean().mul(-1)
        ARP.loc[:,'code'] = code
        ARP.set_index('code', append = True, inplace = True)
        return ARP


class APLFactor(Factor):

    T: int = 19
    back_step: int = 20

    def _call(self, code: str,  daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
        
        df = minutely_df.copy()
        APL = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns=['APL'])
        APL.loc[:, 'APL'] = df.groupby(by = ['date']).tail(self.back_step).groupby(by = ['date']).amount.sum()\
                            .div(df.groupby(by = ['date']).amount.sum()).ewm(span = self.T).mean()
        APL.loc[:,'code'] = code
        APL.set_index('code', append = True, inplace = True)
        return APL


class AMPFactor(Factor):

    T: int = 20
    lam: float = 0.15
    k_freq:int = 60

    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
        
        df = kdata_resample(minutely_df, self.k_freq)
        AMP = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['AMP'])
        df['swing'] = df.high_adj_kfreq.div(df.low_adj_kfreq).sub(1)
        df['group_indexer'] = df.groupby(df.date).ngroup()
        num_of_groups = df.group_indexer.max() + 1
        
        for k in range(self.T, num_of_groups):
            temping = df.loc[(df.group_indexer > (k- self.T)) & (df.group_indexer <= k),:]
            AMP_high = temping.swing[temping.close_adj.rank(pct = True).ge(1 - self.lam)].mean()
            AMP_low = temping.swing[temping.close_adj.rank(pct = True).le(self.lam)].mean()
            date_now = temping.loc[temping.group_indexer == k, 'date'].iat[0]
            AMP.at[date_now, 'AMP'] = np.subtract(AMP_high, AMP_low)
            
        AMP.loc[:,'code'] = code
        AMP.set_index('code', append = True, inplace = True)
        return AMP


class CRAFactor(Factor):

    T: int = 20
    back_step: int = 5

    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
        
        df = minutely_df.copy()
        
        def CRA_generator(data):
            
            x = (abs((data['close_adj'] - data['open_adj']) / data['open_adj'])).dropna()
            y = data.loc[x.index, 'amount']
            if len(x) < 7 or len(y) < 7 or not x.any():
                return None
            return stats.spearmanr(x, y)[0] / stats.spearmanr(x[:-5], y[5:])[0]

        CRA = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['CRA'])
        CRA.loc[:,'CRA'] = df.groupby(df.date).apply(CRA_generator).rolling(self.T).mean().mul(-1)
        CRA.loc[:,'code'] = code
        CRA.set_index('code', append = True, inplace = True)
        return CRA



class INLFactor(Factor):

    T: int = 6
    back_step: int = 25

    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
        
        df = minutely_df.copy()
        INL = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['INL'])
#        df.loc[:,'amount_5mins'] = df.amount.groupby(df.date, group_keys = False).rolling(5).sum().reset_index(drop = True)
        df.loc[:,'amount_5mins'] = df.loc[:,'amount']\
                .groupby(df.date, group_keys = False).apply(lambda x:x.rolling(5).sum())
        df = df[df.datetime.dt.minute.sub(30) % 5 == 0]
        INL.loc[:,'INL'] = (df.close_adj.div(df.close_adj.groupby(df.date).shift(1)).sub(1).abs().add(1)).groupby(df.date).head(self.T)\
        .groupby(df.date).prod().apply(np.log).div(df.amount_5mins.groupby(df.date).head(self.T).groupby(df.date).sum())\
            .rolling(self.back_step).mean().mul(-1)
        INL.loc[:,'code'] = code
        INL.set_index('code', append = True, inplace = True)
        return INL


class SMTFactor(Factor):

    T: int = 28
    beta: float = 0.32
    volume_pct: float = 0.20

    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
       
        df = minutely_df.copy()
        SMT = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['SMT'])
        df['s'] = df.pctchange.abs().div(df.volume.pow(self.beta))
        VWAP_all = df.close_adj.mul(df.volume).groupby(df.date).sum().rolling(self.T).sum()\
        .div(df.volume.groupby(df.date).sum().rolling(self.T).sum())
        df['group_indexer'] = df.groupby('date').ngroup()
        num_of_groups = df.group_indexer.max() + 1
        
        for k in range(self.T, num_of_groups):
            period = df.loc[(df.group_indexer > (k - self.T)) & (df.group_indexer <= k),:] 
            period = period.sort_values('s', axis = 0, ascending = False)
            period['pctv'] = period.volume.cumsum().div(period.volume.sum())
            period_s = period[period.pctv < self.volume_pct]
            VWAP_s = np.divide(period_s.close_adj.mul(period_s.volume).sum(), period_s.volume.sum())           
            date_now = df.loc[df.group_indexer == k, 'date'].iat[0]
            SMT.at[date_now, 'SMT'] = np.divide(VWAP_s, VWAP_all.at[date_now])
            
        SMT.loc[:,'code'] = code
        SMT.set_index('code', append = True, inplace = True)
        return SMT


class WKTFactor(Factor):

    T: int = 20
    back_step: int = 30

    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
      
        df = minutely_df.copy()
        WKT = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['WKT'])
        df['group_indexer'] = df.groupby(df.date).ngroup()
        num_of_groups = df.group_indexer.max() + 1
        
        for k in range(self.T, num_of_groups):
            temp = df[(df.group_indexer > (k - self.T)) & (df.group_indexer <= k)].copy()        
            VOL = temp.volume.sum()
            weighted_kurt = ((temp.volume / VOL).mul(temp.close_adj.sub(temp.close_adj.mean()).pow(3)).sum())/(temp.close_adj.std() ** 3)    
            date_now = df.loc[df.group_indexer == k, 'date'].iat[0]
            WKT.at[date_now,:] = weighted_kurt
            
        WKT.loc[:,'WKT'] = WKT.rolling(self.back_step).mean()
        WKT.loc[:,'code'] = code
        WKT.set_index('code', append = True, inplace = True)
        return WKT


class PVCFactor(Factor):

    T: int = 20
    back_step: int = 30
    
    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):
       
        df = minutely_df.copy()    
        PVC = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['PV_corr_avg','PV_corr_std','PV_corr_trend'])
        df['group_indexer'] = df.groupby(df.date).ngroup()
        num_of_groups = df.group_indexer.max() + 1

        for k in range(self.T, num_of_groups):
            temp = df[(df.group_indexer > (k - self.T)) & (df.group_indexer <= k)].copy()
            date_now = temp.loc[temp.group_indexer == k, 'date'].iat[0]
            pvc_corr = temp.loc[:,['close_adj','volume']].corr().iat[0,1]
            PVC.loc[date_now,:] = [pvc_corr] * 3
        
        PVC.loc[:,'PV_corr_trend'] = PVC.loc[:,'PV_corr_trend'].rolling(self.back_step)\
            .apply(func = lambda x: np.dot(np.arange(1, self.back_step + 1),x)/np.sum(np.arange(1, self.back_step+1)**2), raw = True)
        #目前先用PV_corr_trend
        PVC.loc[:,['PV_corr_avg','PV_corr_std']] = PVC.PV_corr_avg.rolling(self.back_step).agg(['mean','std']).to_numpy()
        PVC = ((PVC.PV_corr_trend - PVC.PV_corr_trend.rolling(self.back_step).mean())/(PVC.PV_corr_trend.rolling(self.back_step).std())).to_frame('PVC')
        PVC.loc[:,'code'] = code
        PVC.set_index('code', append = True, inplace = True)
        return PVC


class UIDFactor(Factor):
    ''''''
    T1:int = 20
    T2:int = 20
    k_freq:int = 5
         
    def _call(self, code: str, daily_df: pd.DataFrame, minutely_df: pd.DataFrame):

        df = kdata_resample(minutely_df, self.k_freq)
        df = df[df.datetime.dt.minute.sub(30) % self.k_freq == 0]

        UID = pd.DataFrame(np.nan, index = df.date.drop_duplicates(), columns = ['UID'])
        vol_daily = df.pctchange.groupby(df.date).std()
        UID = (vol_daily.rolling(self.T1).std() / vol_daily.rolling(self.T2).mean()).to_frame(name = 'UID')
        UID['code'] = code
        UID.set_index('code', append = True, inplace = True)
        return UID

