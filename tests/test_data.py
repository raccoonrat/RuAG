# tests/test_data.py
import pytest
from src.data.preprocess import preprocess_data

def test_preprocess():
    assert preprocess_data("data/raw/test.csv", "data/processed/test.csv") is not None

