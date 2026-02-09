"""PyCaret MCP Server - Main entry point with MCP tool definitions."""
import logging
import traceback
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

# Handle both module execution and direct file execution
try:
    # When run as module: python -m pycaret_mcp_server.server
    from .core.config import (
        mcp, LOG_LEVEL, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT
    )
    from .core.data_loader import load_data, get_data_summary
    from .core import classification as clf
    from .core import regression as reg
except ImportError:
    # When run directly: mcp dev pycaret_mcp_server/server.py
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pycaret_mcp_server.core.config import (
        mcp, LOG_LEVEL, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT
    )
    from pycaret_mcp_server.core.data_loader import load_data, get_data_summary
    from pycaret_mcp_server.core import classification as clf
    from pycaret_mcp_server.core import regression as reg


def setup_logging():
    """Configure logging with all components writing to a single file."""
    # Create logs directory from config
    log_dir = os.path.dirname(LOG_FILE)
    os.makedirs(log_dir, exist_ok=True)

    # Common formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Configure single rotating file handler using config values
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # Configure root logger using config log level
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        handlers=[
            file_handler,
            logging.StreamHandler()
        ]
    )

    return {'server': LOG_FILE}


logger = logging.getLogger(__name__)


def init_logging():
    """Initialize logging system and verify setup."""
    try:
        log_files = setup_logging()
        logger.info(f"Logging configured with single log file: {list(log_files.values())[0]}")
        return True
    except PermissionError as e:
        logger.error(f"Failed to create/access log files: {e}")
        return False


# =============================================================================
# MCP TOOLS - Data Loading
# =============================================================================

@mcp.tool()
def load_dataset_tool(file_path: str, sheet_name: Optional[str] = None) -> dict:
    """Load dataset from CSV or Excel file and return metadata.
    
    Args:
        file_path: Absolute path to data file (CSV, XLSX, XLS)
        sheet_name: Sheet name for Excel files (optional, defaults to first sheet)
    
    Returns:
        dict: Dataset metadata including:
            - status: SUCCESS/ERROR
            - metadata: file info, shape, columns, dtypes
            - summary: basic statistics of the data
    """
    try:
        logger.info(f"Loading dataset: {file_path}")
        result = load_data(file_path, sheet_name)
        
        if result['status'] == 'SUCCESS':
            summary = get_data_summary(result['data'])
            return {
                'status': 'SUCCESS',
                'metadata': result['metadata'],
                'summary': summary,
                'message': f"Loaded {result['metadata']['rows']} rows x {result['metadata']['columns']} columns"
            }
        return result
        
    except Exception as e:
        logger.error(f"load_dataset_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


# =============================================================================
# MCP TOOLS - Classification
# =============================================================================

@mcp.tool()
def setup_classification_tool(
    file_path: str, 
    target: str,
    train_size: float = 0.7,
    session_id: int = 42
) -> dict:
    """Setup PyCaret classification experiment.
    
    Args:
        file_path: Absolute path to data file
        target: Name of target column for classification
        train_size: Proportion of data for training (default: 0.7)
        session_id: Random seed for reproducibility (default: 42)
    
    Returns:
        dict: Setup result with experiment info
    """
    try:
        logger.info(f"Setting up classification for: {file_path}, target: {target}")
        
        data_result = load_data(file_path)
        if data_result['status'] != 'SUCCESS':
            return data_result
        
        return clf.setup_classification(
            data=data_result['data'],
            target=target,
            train_size=train_size,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"setup_classification_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def compare_classification_models_tool(
    fold: int = 5,
    sort: str = "Accuracy",
    n_select: int = 3
) -> dict:
    """Compare all classification models and return best performers.
    
    Args:
        fold: Number of cross-validation folds (default: 5)
        sort: Metric to sort by - Accuracy, AUC, Recall, Precision, F1 (default: Accuracy)
        n_select: Number of top models to return (default: 3)
    
    Returns:
        dict: Model comparison results with rankings
    """
    try:
        logger.info(f"Comparing classification models, sort by: {sort}")
        return clf.compare_models(fold=fold, sort=sort, n_select=n_select)
    except Exception as e:
        logger.error(f"compare_classification_models_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def create_classification_model_tool(model_name: str) -> dict:
    """Create and train a specific classification model.
    
    Args:
        model_name: Model abbreviation. Available options:
            - lr: Logistic Regression
            - knn: K Neighbors Classifier
            - nb: Naive Bayes
            - dt: Decision Tree
            - rf: Random Forest
            - xgboost: XGBoost
            - lightgbm: LightGBM
            - catboost: CatBoost
    
    Returns:
        dict: Created model information
    """
    try:
        logger.info(f"Creating classification model: {model_name}")
        return clf.create_model(model_name)
    except Exception as e:
        logger.error(f"create_classification_model_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def tune_classification_model_tool(
    n_iter: int = 10,
    optimize: str = "Accuracy"
) -> dict:
    """Tune hyperparameters of the current classification model.
    
    Args:
        n_iter: Number of iterations for tuning (default: 10)
        optimize: Metric to optimize - Accuracy, AUC, Recall, Precision, F1 (default: Accuracy)
    
    Returns:
        dict: Tuned model information
    """
    try:
        logger.info(f"Tuning classification model, optimize: {optimize}")
        return clf.tune_model(n_iter=n_iter, optimize=optimize)
    except Exception as e:
        logger.error(f"tune_classification_model_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def predict_classification_tool(file_path: str) -> dict:
    """Make predictions using the trained classification model.
    
    Args:
        file_path: Absolute path to data file for prediction
    
    Returns:
        dict: Prediction results with sample predictions
    """
    try:
        logger.info(f"Making classification predictions for: {file_path}")
        
        data_result = load_data(file_path)
        if data_result['status'] != 'SUCCESS':
            return data_result
        
        return clf.predict_model(data_result['data'])
        
    except Exception as e:
        logger.error(f"predict_classification_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def save_classification_model_tool(model_path: str) -> dict:
    """Save the trained classification model to disk.
    
    Args:
        model_path: Path to save the model (without .pkl extension)
    
    Returns:
        dict: Save result with file path
    """
    try:
        logger.info(f"Saving classification model to: {model_path}")
        return clf.save_model(model_path)
    except Exception as e:
        logger.error(f"save_classification_model_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


# =============================================================================
# MCP TOOLS - Regression
# =============================================================================

@mcp.tool()
def setup_regression_tool(
    file_path: str, 
    target: str,
    train_size: float = 0.7,
    session_id: int = 42
) -> dict:
    """Setup PyCaret regression experiment.
    
    Args:
        file_path: Absolute path to data file
        target: Name of target column for regression (numeric)
        train_size: Proportion of data for training (default: 0.7)
        session_id: Random seed for reproducibility (default: 42)
    
    Returns:
        dict: Setup result with experiment info
    """
    try:
        logger.info(f"Setting up regression for: {file_path}, target: {target}")
        
        data_result = load_data(file_path)
        if data_result['status'] != 'SUCCESS':
            return data_result
        
        return reg.setup_regression(
            data=data_result['data'],
            target=target,
            train_size=train_size,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"setup_regression_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def compare_regression_models_tool(
    fold: int = 5,
    sort: str = "R2",
    n_select: int = 3
) -> dict:
    """Compare all regression models and return best performers.
    
    Args:
        fold: Number of cross-validation folds (default: 5)
        sort: Metric to sort by - MAE, MSE, RMSE, R2, RMSLE, MAPE (default: R2)
        n_select: Number of top models to return (default: 3)
    
    Returns:
        dict: Model comparison results with rankings
    """
    try:
        logger.info(f"Comparing regression models, sort by: {sort}")
        return reg.compare_models(fold=fold, sort=sort, n_select=n_select)
    except Exception as e:
        logger.error(f"compare_regression_models_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def create_regression_model_tool(model_name: str) -> dict:
    """Create and train a specific regression model.
    
    Args:
        model_name: Model abbreviation. Available options:
            - lr: Linear Regression
            - lasso: Lasso Regression
            - ridge: Ridge Regression
            - rf: Random Forest
            - xgboost: XGBoost
            - lightgbm: LightGBM
            - catboost: CatBoost
    
    Returns:
        dict: Created model information
    """
    try:
        logger.info(f"Creating regression model: {model_name}")
        return reg.create_model(model_name)
    except Exception as e:
        logger.error(f"create_regression_model_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def tune_regression_model_tool(
    n_iter: int = 10,
    optimize: str = "R2"
) -> dict:
    """Tune hyperparameters of the current regression model.
    
    Args:
        n_iter: Number of iterations for tuning (default: 10)
        optimize: Metric to optimize - MAE, MSE, RMSE, R2 (default: R2)
    
    Returns:
        dict: Tuned model information
    """
    try:
        logger.info(f"Tuning regression model, optimize: {optimize}")
        return reg.tune_model(n_iter=n_iter, optimize=optimize)
    except Exception as e:
        logger.error(f"tune_regression_model_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def predict_regression_tool(file_path: str) -> dict:
    """Make predictions using the trained regression model.
    
    Args:
        file_path: Absolute path to data file for prediction
    
    Returns:
        dict: Prediction results with statistics
    """
    try:
        logger.info(f"Making regression predictions for: {file_path}")
        
        data_result = load_data(file_path)
        if data_result['status'] != 'SUCCESS':
            return data_result
        
        return reg.predict_model(data_result['data'])
        
    except Exception as e:
        logger.error(f"predict_regression_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


@mcp.tool()
def save_regression_model_tool(model_path: str) -> dict:
    """Save the trained regression model to disk.
    
    Args:
        model_path: Path to save the model (without .pkl extension)
    
    Returns:
        dict: Save result with file path
    """
    try:
        logger.info(f"Saving regression model to: {model_path}")
        return reg.save_model(model_path)
    except Exception as e:
        logger.error(f"save_regression_model_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


# =============================================================================
# MCP TOOLS - Utility
# =============================================================================

@mcp.tool()
def get_available_models_tool(task_type: str = "classification") -> dict:
    """Get list of available models for a given task type.
    
    Args:
        task_type: Either 'classification' or 'regression'
    
    Returns:
        dict: Available model names and descriptions
    """
    try:
        if task_type.lower() == 'classification':
            return clf.get_available_models()
        elif task_type.lower() == 'regression':
            return reg.get_available_models()
        else:
            return {
                'status': 'ERROR',
                'message': f"Unknown task type: {task_type}. Use 'classification' or 'regression'."
            }
    except Exception as e:
        logger.error(f"get_available_models_tool failed: {str(e)}")
        return {'status': 'ERROR', 'message': str(e)}


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for the MCP server."""
    try:
        if not init_logging():
            raise RuntimeError("Failed to initialize logging")

        logger.info("Starting PyCaret MCP Server...")
        mcp.run()
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
