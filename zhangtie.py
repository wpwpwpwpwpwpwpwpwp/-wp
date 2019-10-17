# coding=utf8
__author__ = 'wangjp'

import time

import numpy as np
import pandas as pd
from FactorModule.FactorBase import FactorBase
from DataReaderModule.Constants import ALIAS_FIELDS as t

class Factor(FactorBase):

    def __init__(self):
        super(Factor,self).__init__()
        self.neutral = False
        self.factorName = __name__.split('.')[-1]
        self.needFields = [t.HIGH, t.LOW, t.CLOSE, t.ADJFCT, t.VOLUME,t.TRDSTAT]  # 设置需要的字段
    def dailyVariationRate(x):
        ColNames = x.columns
        for i in range(len(ColNames)):
            x[ColNames[i]]=x[ColNames[i]].pct_change()
        return x
    def RisingAndStoppingDays(x,x1,nums,k):
        y=Factor.dailyVariationRate(x1)
        M=x.shape[0]
        ColNames = x.columns
        for i in range(len(ColNames)):
           for j in range(nums):
               x[ColNames[i]].values[j]=None
           for j in range(nums,M):
               n=0
               for jj in range(j-nums,j):
                   if y[ColNames[i]].values[j]>0.098:
                       n=n+1
               if n<k:
                   x[ColNames[i]].values[j]=None
        return x
    def factor_definition(self):
        """
        收集派发指标
        :return:
        """
        s = time.time()
        needData = self.needData                                # 计算所需数据
        adjVolume=needData[t.VOLUME]
        adjHigh = needData[t.HIGH] * needData[t.ADJFCT]
        adjClose = needData[t.CLOSE] * needData[t.ADJFCT]
        #factor = needData[t.CLOSE]
        a=Factor.RisingAndStoppingDays(adjClose,adjClose,100,5)
        #a=Factor.dailyVariationRate(adjClose)
        factor=self.calculator.Corr(a,adjVolume,10)
        #factor=self.calculator.Corr(adjClose,adjHigh,10)
        print(111111111111,factor)
        print('factor {0} done with {1} seconds'.format(self.factorName, time.time() - s))
        return factor

    def run_factor(self):
        self.run()



fct = Factor()
fct.run_factor()