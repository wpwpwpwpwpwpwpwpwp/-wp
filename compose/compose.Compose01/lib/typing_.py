from enum import Enum
from typing import *

import mltk

__all__ = [
    'ComposeMethod',

    'ModelConfig',
    'ModelSelectedFeature',
    'ModelParams',
    'Model',
]


class ComposeMethod(str, Enum):
    IC_TOP_K = 'ic_top_k'
    xxxx = 'xxxx'


class ComposeConfig(str, Enum):
    method: ComposeMethod

    # whitelist feature names
    whitelist_features: List[str]   # jaysus.feature.Bundle03/main/Tech_*

    # configs for "TOP_K"
    num_features: int = 50


class ModelConfig(mltk.Config):
    compose: ComposeConfig


class ModelSelectedFeature(mltk.Config):
    __slots__ = ['name', 'weight']

    name: str
    weight: float


class ModelParams(mltk.Config):

    features: List[ModelSelectedFeature]


class Model(mltk.Config):

    config: ModelConfig
    params: ModelParams

    @classmethod
    def load_file(cls, path: str):
        loader = mltk.ConfigLoader(Model)
        loader.load_file(path)
        return loader.get()
