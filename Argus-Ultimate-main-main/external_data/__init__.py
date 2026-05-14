"""
External Data Sources - Free APIs for constant improvement.

Fetches derivatives data, sentiment, whale alerts, and macro economic data
from multiple free APIs.
"""

from external_data.free_data_fetcher import FreeDataFetcher, get_free_data_fetcher

__all__ = ["FreeDataFetcher", "get_free_data_fetcher"]
