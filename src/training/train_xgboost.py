import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Enhanced XGBoost training for OLINK COVID-19 classification"""
    
    parser = argparse.ArgumentParser(description='XGBoost training for OLINK data')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Hyperparameters (from your existing config)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.3)
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--eval_metric', type=str, default='auc')
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--early_stopping_rounds', type=int, default=10)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    
    # Additional arguments
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=1)
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting XGBoost training for OLINK COVID-19 classification")
    logger.info(f"📁 Model directory: {args.model_dir}")
    logger.info(f"📊 Training data: {args.train}")
    logger.info(f"📈 Validation data: {args.validation}")
    
    try:
        # FIXED: Load training and validation data without headers
        logger.info("📖 Loading training data...")
        train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
        if not train_files:
            raise FileNotFoundError(f"No CSV files found in {args.train}")
        
        train_df = pd.read_csv(os.path.join(args.train, train_files[0]), header=None)
        
        logger.info("📖 Loading validation data...")
        val_files = [f for f in os.listdir(args.validation) if f.endswith('.csv')]
        if not val_files:
            raise FileNotFoundError(f"No CSV files found in {args.validation}")
            
        val_df = pd.read_csv(os.path.join(args.validation, val_files[0]), header=None)
        
        logger.info(f"📊 Training data shape: {train_df.shape}")
        logger.info(f"📈 Validation data shape: {val_df.shape}")
        
        # FIXED: Handle data without headers (SageMaker standard format)
        # First column is target, rest are features
        X_train = train_df.iloc[:, 1:]  # All columns except first
        y_train = train_df.iloc[:, 0]   # First column is target
        X_val = val_df.iloc[:, 1:]
        y_val = val_df.iloc[:, 0]
        
        logger.info(f"🎯 Target distribution in training: {y_train.value_counts().to_dict()}")
        logger.info(f"🎯 Target distribution in validation: {y_val.value_counts().to_dict()}")
        
        # Create DMatrix for XGBoost (optimized for memory)
        logger.info("🔄 Creating DMatrix objects...")
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set XGBoost parameters
        params = {
            'max_depth': args.max_depth,
            'eta': args.eta,
            'objective': args.objective,
            'eval_metric': args.eval_metric,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'seed': args.random_state,
            'verbosity': args.verbose
        }
        
        logger.info(f"⚙️ XGBoost parameters: {params}")
        
        # Training with early stopping
        logger.info("🏋️ Starting XGBoost training...")
        evals = [(dtrain, 'train'), (dval, 'validation')]
        
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=args.num_round,
            evals=evals,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=True
        )
        
        logger.info("✅ Training completed!")
        
        # Make predictions on validation set
        logger.info("🔮 Making predictions on validation set...")
        y_val_pred_prob = model.predict(dval)
        y_val_pred = (y_val_pred_prob > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_pred_prob)
        
        logger.info(f"📊 Validation Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Validation F1-Score: {f1:.4f}")
        logger.info(f"📊 Validation AUC: {auc:.4f}")
        
        # FIXED: Feature importance handling for numeric feature names
        feature_importance = model.get_score(importance_type='weight')
        
        # Handle numeric feature names (since we don't have column names)
        feature_importance_data = []
        for feature_name, importance in feature_importance.items():
            feature_importance_data.append({
                'feature': feature_name,
                'importance': importance
            })
        
        feature_importance_df = pd.DataFrame(feature_importance_data).sort_values('importance', ascending=False)
        
        # Prepare results (similar to your existing structure)
        results = {
            'model_type': 'XGBoost',
            'training_timestamp': datetime.now().isoformat(),
            'hyperparameters': params,
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'auc': float(auc)
            },
            'feature_importance_top10': feature_importance_df.head(10).to_dict('records'),
            'training_info': {
                'num_features': X_train.shape[1],  # FIXED: Use shape instead of len(columns)
                'num_training_samples': len(X_train),
                'num_validation_samples': len(X_val),
                'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else args.num_round
            }
        }
        
        # FIXED: Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Save model
        logger.info("💾 Saving model...")
        model_path = os.path.join(args.model_dir, 'xgboost-model')
        model.save_model(model_path)
        
        # Save model as pickle for compatibility (like your existing code)
        joblib.dump(model, os.path.join(args.model_dir, 'XGBoost_model.pkl'))
        
        # Save metrics and results
        with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature importance (like your existing output)
        feature_importance_df.to_csv(
            os.path.join(args.model_dir, 'XGBoost_feature_importance.csv'), 
            index=False
        )
        
        # Save classification report
        class_report = classification_report(y_val, y_val_pred, output_dict=True)
        with open(os.path.join(args.model_dir, 'classification_report.json'), 'w') as f:
            json.dump(class_report, f, indent=2)
        
        # Save confusion matrix
        cm = confusion_matrix(y_val, y_val_pred)
        np.save(os.path.join(args.model_dir, 'confusion_matrix.npy'), cm)
        
        logger.info("✅ Model training completed successfully!")
        logger.info(f"📁 Model saved to: {args.model_dir}")
        
        # Print summary (for SageMaker logs)
        print("=" * 50)
        print("XGBOOST TRAINING SUMMARY")
        print("=" * 50)
        print(f"Model Type: XGBoost")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Best Iteration: {model.best_iteration if hasattr(model, 'best_iteration') else args.num_round}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise e

if __name__ == '__main__':
    main()