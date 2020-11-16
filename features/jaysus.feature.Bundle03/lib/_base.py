from logging import getLogger

import pandas as pd

__all__ = ['factors', 'Factor']

factors = []


class FactorMeta(type):

    def __new__(cls, name, parents, dct):
        if name in factors:
            raise ValueError(f'Duplicated factor name: {name!r}')
        ret_cls = super(FactorMeta, cls).__new__(cls, name, parents, dct)
        if name != 'Factor':
            factors.append(ret_cls())
        return ret_cls


class Factor(metaclass=FactorMeta):
    
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _call(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()

    def __call__(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        try:
            return self._call(daily_df)
        except NotImplementedError:
            getLogger(__name__).error('Factor not implemented: %r', self)
            raise
