"""Classification module for pycaret-mcp-server using PyCaret."""
import logging
import os
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

# Global state for classification experiment
_classification_setup = None
_classification_model = None


def setup_classification(
    data: pd.DataFrame, 
    target: str,
    session_id: int = 42,
    train_size: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Setup PyCaret classification experiment.
    
    Args:
        data: pandas DataFrame with features and target
        target: name of target column
        session_id: random seed for reproducibility
        train_size: proportion of data for training
        **kwargs: additional PyCaret setup parameters
    
    Returns:
        dict: Setup result with status and info
    """
    global _classification_setup
    
    try:
        from pycaret.classification import setup, get_config
        
        logger.info(f"Setting up classification with target: {target}")
        
        # Validate target column exists
        if target not in data.columns:
            return {
                'status': 'ERROR',
                'message': f"Target column '{target}' not found in data. Available columns: {data.columns.tolist()}"
            }
        
        # Setup PyCaret classification
        _classification_setup = setup(
            data=data,
            target=target,
            session_id=session_id,
            train_size=train_size,
            verbose=False,
            html=False,
            **kwargs
        )
        
        # Get setup info
        setup_info = {
            'target': target,
            'target_type': str(data[target].dtype),
            'n_classes': int(data[target].nunique()),
            'class_distribution': data[target].value_counts().to_dict(),
            'train_size': train_size,
            'session_id': session_id,
            'n_features': len(data.columns) - 1,
            'n_samples': len(data)
        }
        
        logger.info(f"Classification setup complete: {setup_info['n_classes']} classes, {setup_info['n_features']} features")
        
        return {
            'status': 'SUCCESS',
            'message': 'Classification experiment setup complete',
            'setup_info': setup_info
        }
        
    except Exception as e:
        logger.error(f"Classification setup failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }


def compare_models(
    fold: int = 5,
    sort: str = 'Accuracy',
    n_select: int = 3
) -> Dict[str, Any]:
    """
    Compare all classification models and return best performers.
    
    Args:
        fold: number of cross-validation folds
        sort: metric to sort by
        n_select: number of top models to return
    
    Returns:
        dict: Comparison results with model rankings
    """
    try:
        from pycaret.classification import compare_models as pc_compare
        
        logger.info(f"Comparing classification models with {fold} folds, sorting by {sort}")
        
        best_models = pc_compare(fold=fold, sort=sort, n_select=n_select, verbose=False)
        
        # Handle single model vs list
        if not isinstance(best_models, list):
            best_models = [best_models]
        
        results = []
        for i, model in enumerate(best_models):
            model_name = type(model).__name__
            results.append({
                'rank': i + 1,
                'model_name': model_name,
                'model_type': str(type(model))
            })
        
        logger.info(f"Top models: {[r['model_name'] for r in results]}")
        
        return {
            'status': 'SUCCESS',
            'message': f'Compared models successfully. Top {len(results)} models returned.',
            'best_models': results,
            'sort_metric': sort
        }
        
    except Exception as e:
        logger.error(f"Model comparison failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }


def create_model(model_name: str, **kwargs) -> Dict[str, Any]:
    """
    Create and train a specific classification model.
    
    Args:
        model_name: PyCaret model abbreviation (e.g., 'lr', 'rf', 'xgboost')
        **kwargs: additional model parameters
    
    Returns:
        dict: Created model info
    """
    global _classification_model
    
    try:
        from pycaret.classification import create_model as pc_create
        
        logger.info(f"Creating classification model: {model_name}")
        
        _classification_model = pc_create(model_name, verbose=False, **kwargs)
        
        model_info = {
            'model_name': model_name,
            'model_type': type(_classification_model).__name__,
            'model_params': str(_classification_model.get_params()) if hasattr(_classification_model, 'get_params') else 'N/A'
        }
        
        logger.info(f"Model created: {model_info['model_type']}")
        
        return {
            'status': 'SUCCESS',
            'message': f'Model {model_name} created successfully',
            'model_info': model_info
        }
        
    except Exception as e:
        logger.error(f"Model creation failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }


def tune_model(
    n_iter: int = 10,
    optimize: str = 'Accuracy',
    **kwargs
) -> Dict[str, Any]:
    """
    Tune hyperparameters of the current model.
    
    Args:
        n_iter: number of iterations for tuning
        optimize: metric to optimize
        **kwargs: additional tuning parameters
    
    Returns:
        dict: Tuned model info
    """
    global _classification_model
    
    try:
        from pycaret.classification import tune_model as pc_tune
        
        if _classification_model is None:
            return {
                'status': 'ERROR',
                'message': 'No model to tune. Create a model first using create_model.'
            }
        
        logger.info(f"Tuning model with {n_iter} iterations, optimizing {optimize}")
        
        _classification_model = pc_tune(
            _classification_model,
            n_iter=n_iter,
            optimize=optimize,
            verbose=False,
            **kwargs
        )
        
        return {
            'status': 'SUCCESS',
            'message': f'Model tuned successfully',
            'tuned_model_type': type(_classification_model).__name__
        }
        
    except Exception as e:
        logger.error(f"Model tuning failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }


def predict_model(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Make predictions using the trained model.
    
    Args:
        data: DataFrame with features for prediction
    
    Returns:
        dict: Predictions result
    """
    global _classification_model
    
    try:
        from pycaret.classification import predict_model as pc_predict
        
        if _classification_model is None:
            return {
                'status': 'ERROR',
                'message': 'No trained model available. Create a model first.'
            }
        
        logger.info(f"Making predictions on {len(data)} samples")
        
        predictions = pc_predict(_classification_model, data=data)
        
        # Get prediction columns
        pred_cols = [col for col in predictions.columns if 'prediction' in col.lower() or 'label' in col.lower()]
        
        return {
            'status': 'SUCCESS',
            'message': f'Predictions made for {len(predictions)} samples',
            'n_predictions': len(predictions),
            'prediction_columns': pred_cols,
            'predictions_sample': predictions[pred_cols].head(10).to_dict() if pred_cols else {}
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }


def save_model(model_path: str) -> Dict[str, Any]:
    """
    Save the trained model to disk.
    
    Args:
        model_path: path to save the model (without extension)
    
    Returns:
        dict: Save result
    """
    global _classification_model
    
    try:
        from pycaret.classification import save_model as pc_save
        
        if _classification_model is None:
            return {
                'status': 'ERROR',
                'message': 'No trained model to save.'
            }
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        pc_save(_classification_model, model_path, verbose=False)
        
        logger.info(f"Model saved to: {model_path}")
        
        return {
            'status': 'SUCCESS',
            'message': f'Model saved to {model_path}.pkl',
            'model_path': f'{model_path}.pkl'
        }
        
    except Exception as e:
        logger.error(f"Model save failed: {str(e)}")
        return {
            'status': 'ERROR',
            'message': str(e)
        }


def get_available_models() -> Dict[str, Any]:
    """
    Get list of available classification models in PyCaret.
    
    Returns:
        dict: Available model names and descriptions
    """
    models = {
        'lr': 'Logistic Regression',
        'knn': 'K Neighbors Classifier',
        'nb': 'Naive Bayes',
        'dt': 'Decision Tree Classifier',
        'svm': 'SVM - Linear Kernel',
        'rbfsvm': 'SVM - Radial Kernel',
        'gpc': 'Gaussian Process Classifier',
        'mlp': 'MLP Classifier',
        'ridge': 'Ridge Classifier',
        'rf': 'Random Forest Classifier',
        'qda': 'Quadratic Discriminant Analysis',
        'ada': 'Ada Boost Classifier',
        'gbc': 'Gradient Boosting Classifier',
        'lda': 'Linear Discriminant Analysis',
        'et': 'Extra Trees Classifier',
        'xgboost': 'Extreme Gradient Boosting',
        'lightgbm': 'Light Gradient Boosting Machine',
        'catboost': 'CatBoost Classifier'
    }
    
    return {
        'status': 'SUCCESS',
        'available_models': models
    }
