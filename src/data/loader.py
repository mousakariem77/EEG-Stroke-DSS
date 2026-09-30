"""
Module M1: Data Loading & Validation

Loads the EPoC dataset (Aminov et al., 2017) and prepares it for the pipeline.
The dataset contains PRE-COMPUTED EEG features (not raw EEG signals).

Dataset details:
- Source: Dryad — doi:10.5061/dryad.h6986
- 38 rows: 19 stroke + 19 control (balanced)
- 29 columns: EEG features (pre-computed) + demographics + clinical data
- Single electrode: FP1 (pre-frontal)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


# EEG features available in the dataset (pre-computed from PSD analysis)
EEG_FEATURES = [
    'Delta', 'Theta', 'Alpha', 'Beta',          # Absolute power per band
    'RP Delta', 'RP Theta', 'RP Alpha', 'RP Beta',  # Relative power per band
    'DTR', 'DAR',                                 # Delta/Theta ratio, Delta/Alpha ratio
    'Total Power', 'Epoch'                        # Total power and epoch count
]

# Demographic features
DEMOGRAPHIC_FEATURES = ['Age', 'Gender', 'Years of education']

# Features used for classification (based on SHAP analysis in paper)
# Paper's top 8: DTR, Theta, RP Delta, RP Alpha, age, beta, years of education, gender
PAPER_TOP_FEATURES = [
    'DTR', 'RP Delta', 'RP Theta', 'RP Alpha', 'RP Beta',
    'Age', 'Gender', 'Years of education', 'Beta'
]

# Target column
TARGET_COLUMN = 'Participant'

# Label encoding
LABEL_MAP = {'control': 0, 'stroke': 1}
LABEL_MAP_INV = {0: 'control', 1: 'stroke'}


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """
    Load the EPoC dataset from Excel file.
    
    Args:
        data_path: Path to the xlsx file. If None, auto-detect from data/raw/
    
    Returns:
        pd.DataFrame: Raw dataset
    """
    if data_path is None:
        # Auto-detect from data/raw/
        raw_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        xlsx_files = list(raw_dir.glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(
                f"No .xlsx files found in {raw_dir}. "
                "Please download EPoC Data from https://doi.org/10.5061/dryad.h6986"
            )
        data_path = xlsx_files[0]
    
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    df = pd.read_excel(data_path, engine='openpyxl')
    print(f"[DATA] Loaded dataset from {data_path.name}: {df.shape[0]} rows x {df.shape[1]} columns")
    
    return df


def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Validate the dataset structure and report any issues.
    
    Args:
        df: Raw dataset DataFrame
    
    Returns:
        dict: Validation report
    """
    report = {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'issues': [],
        'warnings': []
    }
    
    # Check target column exists
    if TARGET_COLUMN not in df.columns:
        report['issues'].append(f"Target column '{TARGET_COLUMN}' not found")
    else:
        classes = df[TARGET_COLUMN].unique()
        report['classes'] = list(classes)
        report['class_counts'] = df[TARGET_COLUMN].value_counts().to_dict()
    
    # Check EEG features exist
    missing_eeg = [f for f in EEG_FEATURES if f not in df.columns]
    if missing_eeg:
        report['issues'].append(f"Missing EEG features: {missing_eeg}")
    
    # Check demographic features exist
    missing_demo = [f for f in DEMOGRAPHIC_FEATURES if f not in df.columns]
    if missing_demo:
        report['warnings'].append(f"Missing demographic features: {missing_demo}")
    
    # Check for NaN in EEG features
    eeg_cols = [f for f in EEG_FEATURES if f in df.columns]
    nan_counts = df[eeg_cols].isnull().sum()
    if nan_counts.any():
        report['warnings'].append(f"NaN values in EEG features: {nan_counts[nan_counts > 0].to_dict()}")
    
    # Check class balance
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        ratio = counts.min() / counts.max()
        if ratio < 0.5:
            report['warnings'].append(f"Imbalanced classes: ratio={ratio:.2f}")
    
    is_valid = len(report['issues']) == 0
    report['is_valid'] = is_valid
    
    return report


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features (Age, Gender) to numeric.
    
    Age: '<60' -> 0, '>60' -> 1
    Gender: 'female' -> 0, 'male' -> 1
    
    Args:
        df: DataFrame with categorical features
    
    Returns:
        pd.DataFrame: DataFrame with encoded features
    """
    df = df.copy()
    
    # Encode Age: '<60' -> 0, '>60' -> 1
    if 'Age' in df.columns and df['Age'].dtype == 'object':
        df['Age'] = df['Age'].map({'<60': 0, '>60': 1}).astype(int)
        print("[DATA] Encoded Age: <60 -> 0, >60 -> 1")
    
    # Encode Gender: 'female' -> 0, 'male' -> 1
    if 'Gender' in df.columns and df['Gender'].dtype == 'object':
        df['Gender'] = df['Gender'].map({'female': 0, 'male': 1}).astype(int)
        print("[DATA] Encoded Gender: female -> 0, male -> 1")
    
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target labels: 'control' -> 0, 'stroke' -> 1.
    
    Args:
        df: DataFrame with 'Participant' column
    
    Returns:
        pd.DataFrame: DataFrame with encoded target
    """
    df = df.copy()
    
    if TARGET_COLUMN in df.columns:
        df['label'] = df[TARGET_COLUMN].map(LABEL_MAP).astype(int)
        print(f"[DATA] Encoded labels: {LABEL_MAP}")
    
    return df


def prepare_features(
    df: pd.DataFrame,
    feature_set: str = 'all',
    include_demographics: bool = True
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Prepare feature matrix X and label vector y from the dataset.
    
    Args:
        df: DataFrame (after encoding)
        feature_set: 'all' (all EEG + demo), 'eeg_only', 'paper_top' (paper's top features)
        include_demographics: Whether to include Age, Gender, Years of education
    
    Returns:
        Tuple of:
            X: feature matrix (n_samples, n_features)
            y: label vector (n_samples,)
            feature_names: list of feature names
    """
    # Ensure labels are encoded
    if 'label' not in df.columns:
        df = encode_labels(df)
    
    # Ensure categorical features are encoded
    df = encode_categorical_features(df)
    
    # Select features based on feature_set
    if feature_set == 'eeg_only':
        feature_names = [f for f in EEG_FEATURES if f in df.columns]
    elif feature_set == 'paper_top':
        feature_names = [f for f in PAPER_TOP_FEATURES if f in df.columns]
    elif feature_set == 'all':
        feature_names = [f for f in EEG_FEATURES if f in df.columns]
        if include_demographics:
            feature_names += [f for f in DEMOGRAPHIC_FEATURES if f in df.columns]
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}. Use 'all', 'eeg_only', or 'paper_top'")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_features = []
    for f in feature_names:
        if f not in seen:
            seen.add(f)
            unique_features.append(f)
    feature_names = unique_features
    
    X = df[feature_names].values.astype(np.float64)
    y = df['label'].values.astype(np.int64)
    
    print(f"[DATA] Feature matrix: X={X.shape}, y={y.shape}")
    print(f"[DATA] Features ({len(feature_names)}): {feature_names}")
    print(f"[DATA] Class distribution: control={np.sum(y==0)}, stroke={np.sum(y==1)}")
    
    return X, y, feature_names


def load_and_prepare(
    data_path: str = None,
    feature_set: str = 'all',
    include_demographics: bool = True
) -> Tuple[np.ndarray, np.ndarray, list, pd.DataFrame]:
    """
    Convenience function: load, validate, encode, and prepare features.
    
    Returns:
        Tuple of (X, y, feature_names, raw_df)
    """
    # Load
    df = load_dataset(data_path)
    
    # Validate
    report = validate_dataset(df)
    if not report['is_valid']:
        raise ValueError(f"Dataset validation failed: {report['issues']}")
    if report['warnings']:
        for w in report['warnings']:
            print(f"[WARNING] {w}")
    
    # Encode and prepare
    df_encoded = encode_labels(df)
    df_encoded = encode_categorical_features(df_encoded)
    
    X, y, feature_names = prepare_features(df_encoded, feature_set, include_demographics)
    
    return X, y, feature_names, df
