import pandas as pd

__all__ = ['Factor']


class Factor(object):

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def _call(self, code: str, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError()

    def __call__(self, code: str, df: pd.DataFrame) -> pd.Series:
        return self._call(code, df)
