import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.data.preprocessing import load_ml1m_data, preprocess_ratings, split_data

