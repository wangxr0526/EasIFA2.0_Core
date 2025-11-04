"""
EasIFA Core - Enzyme Active Site Inference Framework

This package provides core inference functionality for predicting enzyme active sites.
"""

__version__ = '2.0.0'
__author__ = 'EasIFA Team'

from .config import EasIFAInferenceConfig
from .interface.inference import EasIFAInferenceAPI

__all__ = [
    'EasIFAInferenceConfig',
    'EasIFAInferenceAPI',
]
