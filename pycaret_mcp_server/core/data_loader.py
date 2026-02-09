"""Data loading utilities for pycaret-mcp-server."""
import os
import logging
import pandas as pd
import chardet
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def detect_encoding(file_path: str) -> str:
    """Detect file encoding using chardet."""
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read first 10KB for detection
        result = chardet.detect(raw_data)
        return result.get('encoding', 'utf-8') or 'utf-8'


def load_data(file_path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load data from CSV or Excel file and return as DataFrame with metadata.
    
    Args:
        file_path: Absolute path to data file
        sheet_name: Sheet name for Excel files (optional)
    
    Returns:
        dict: {
            'status': 'SUCCESS' or 'ERROR',
            'data': DataFrame or None,
            'metadata': file info dict,
            'message': error message if any
        }
    """
    try:
        if not os.path.exists(file_path):
            return {
                'status': 'ERROR',
                'data': None,
                'metadata': None,
                'message': f'File not found: {file_path}'
            }
        
        file_ext = os.path.splitext(file_path)[1].lower()
        file_size = os.path.getsize(file_path)
        
        logger.info(f"Loading file: {file_path} ({file_size / 1024 / 1024:.2f} MB)")
        
        if file_ext == '.csv':
            encoding = detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=encoding)
            file_type = 'csv'
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, sheet_name=sheet_name or 0)
            file_type = 'excel'
        else:
            return {
                'status': 'ERROR',
                'data': None,
                'metadata': None,
                'message': f'Unsupported file type: {file_ext}. Supported: .csv, .xlsx, .xls'
            }
        
        metadata = {
            'file_path': file_path,
            'file_type': file_type,
            'file_size_mb': round(file_size / 1024 / 1024, 2),
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'null_counts': df.isnull().sum().to_dict(),
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        logger.info(f"Loaded {metadata['rows']} rows x {metadata['columns']} columns")
        
        return {
            'status': 'SUCCESS',
            'data': df,
            'metadata': metadata,
            'message': None
        }
        
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return {
            'status': 'ERROR',
            'data': None,
            'metadata': None,
            'message': str(e)
        }


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics of a DataFrame.
    
    Args:
        df: pandas DataFrame
    
    Returns:
        dict: Summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
            'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
            'max': float(df[col].max()) if not pd.isna(df[col].max()) else None
        }
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        unique_count = df[col].nunique()
        summary['categorical_summary'][col] = {
            'unique_count': int(unique_count),
            'top_values': df[col].value_counts().head(5).to_dict() if unique_count <= 100 else 'Too many unique values'
        }
    
    return summary
