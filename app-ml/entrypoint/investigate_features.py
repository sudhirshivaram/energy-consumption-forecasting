"""
Investigate feature scales and statistics to understand importance calculation.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)

import pandas as pd
import yaml
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline

# Load config
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load and process data
data_manager = DataManager(config)
raw_data = data_manager.load_raw_data(format='csv')

preprocessing = PreprocessingPipeline(config)
preprocessed_data = preprocessing.run(raw_data)

feature_engineering = FeatureEngineeringPipeline(config)
engineered_data = feature_engineering.run(preprocessed_data)

# Check statistics for the engineered features
print("Feature Statistics:")
print("="*80)
print(engineered_data.describe())

print("\n\nFeature Standard Deviations:")
print("="*80)
for col in engineered_data.columns:
    print(f"{col}: {engineered_data[col].std():.2f}")