from datetime import timedelta

import numpy as np
import pandas as pd

from ._base import *

__all__ = [
    'ARPFactor',
]


class ARPFactor(Factor):

    T: int = 27
    back_step: int = 23

    def _call(self, code: str, df: pd.DataFrame):
        temp = df.loc[df.date == df.date.iat[-1], :].copy()
        temp.loc[:, 'date'] = temp.date.add(timedelta(3))
        temp.loc[:, 'datetime'] = temp.datetime.add(timedelta(3))
        df = pd.concat([df, temp], axis=0, ignore_index=True)

        res = pd.DataFrame(np.nan, index=df.date.drop_duplicates(), columns=['ARP'])
        # 计算因子
        df['group_indexer'] = df.groupby(df.date).ngroup()
        num_of_group = df.group_indexer.max() + 1
        for k in range(self.T, num_of_group):
            temping = df.loc[df.group_indexer.ge(k - self.T + 1) & df.group_indexer.le(k), :]
            rpp = temping.close_adj.sub(temping.low_adj.min()).div(temping.high_adj.max() - temping.low_adj.min())
            arp = rpp.sum()
            date_now = temping.loc[temping.group_indexer == k, 'date'].iat[0]
            res.at[date_now, 'ARP'] = arp
        res.loc[:, 'ARP'] = res.ARP.rolling(self.back_step).mean().mul(-1).shift()
        res['code'] = code
        res.set_index('code', append=True, inplace=True)
        return res['ARP']
