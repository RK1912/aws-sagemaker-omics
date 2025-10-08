import argparse
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
import json
import mlflow
import mlflow.sklearn
import logging
from datetime import datetime
import subprocess

mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", None)
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("OLINK-Experiments")

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """XGBoost training"""
    
    parser = argparse.ArgumentParser(description='XGBoost training for OLINK data')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Hyperparameters
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
    
    # hyperparameters from sagemaker environment variables
    max_depth = int(os.environ.get('SM_HP_MAX_DEPTH', args.max_depth))
    eta = float(os.environ.get('SM_HP_ETA', args.eta))
    num_round = int(os.environ.get('SM_HP_NUM_ROUND', args.num_round))
    
    logger.info("🚀 Starting XGBoost training for OLINK COVID-19 classification")
    logger.info(f"📁 Model directory: {args.model_dir}")
    logger.info(f"📊 Training data: {args.train}")
    logger.info(f"📈 Validation data: {args.validation}")
    logger.info(f"⚙️ Hyperparameters: max_depth={max_depth}, eta={eta}, num_round={num_round}")
    
    try:
        # Load training and validation data without headers
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
        
        # Handle data without headers (SageMaker standard format)
        # First column is target, rest are features
        X_train = train_df.iloc[:, 1:]  # All columns except first
        y_train = train_df.iloc[:, 0]   # First column is target
        X_val = val_df.iloc[:, 1:]
        y_val = val_df.iloc[:, 0]
        
        logger.info(f"🎯 Target distribution in training: {y_train.value_counts().to_dict()}")
        logger.info(f"🎯 Target distribution in validation: {y_val.value_counts().to_dict()}")
        
        # Convert string labels to numeric
        label_encoder = None
        if y_train.dtype == 'object' or y_train.dtype.name == 'string':
            logger.info("🔄 Converting string labels to numeric...")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
            
            # Log the mapping
            label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            logger.info(f"🔢 Label mapping: {label_mapping}")
        else:
            # Ensure labels are numeric
            y_train_encoded = pd.to_numeric(y_train, errors='coerce')
            y_val_encoded = pd.to_numeric(y_val, errors='coerce')
            
            # Check for any conversion issues
            if y_train_encoded.isna().any() or y_val_encoded.isna().any():
                raise ValueError("Unable to convert labels to numeric. Please check your data.")
        
        # ensure features are numeric
        logger.info("🔄 Ensuring features are numeric...")
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_val = X_val.apply(pd.to_numeric, errors='coerce')
        
        # Check for missing values
        if X_train.isna().any().any():
            logger.warning("⚠️ Found NaN values in training features after conversion. Filling with 0.")
            X_train = X_train.fillna(0)
        
        if X_val.isna().any().any():
            logger.warning("⚠️ Found NaN values in validation features after conversion. Filling with 0.")
            X_val = X_val.fillna(0)
        
        # Create DMatrix for XGBoost 
        logger.info("🔄 Creating DMatrix objects...")
        dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
        dval = xgb.DMatrix(X_val, label=y_val_encoded)
        
        # Set XGBoost parameters
        params = {
            'max_depth': max_depth,
            'eta': eta,
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
            num_boost_round=num_round,
            evals=evals,
            early_stopping_rounds=args.early_stopping_rounds,
            verbose_eval=True
        )
        
        logger.info("✅ Training completed!")
        
        # Make predictions on validation set
        logger.info("🔮 Making predictions on validation set...")
        y_val_pred_prob = model.predict(dval)
        y_val_pred = (y_val_pred_prob > 0.5).astype(int)
        
        # Calculate metrics using encoded labels
        accuracy = accuracy_score(y_val_encoded, y_val_pred)
        f1 = f1_score(y_val_encoded, y_val_pred)
        auc = roc_auc_score(y_val_encoded, y_val_pred_prob)
        
        logger.info(f"📊 Validation Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Validation F1-Score: {f1:.4f}")
        logger.info(f"📊 Validation AUC: {auc:.4f}")
        
        # Feature importance handling for numeric feature names
        feature_importance = model.get_score(importance_type='weight')
        
        # Handle numeric feature names
        feature_importance_data = []
        for feature_name, importance in feature_importance.items():
            feature_importance_data.append({
                'feature': feature_name,
                'importance': importance
            })
        
        feature_importance_df = pd.DataFrame(feature_importance_data).sort_values('importance', ascending=False)
        
        # Convert feature importance to JSON-serializable format
        feature_importance_json = []
        for _, row in feature_importance_df.head(10).iterrows():
            feature_importance_json.append({
                'feature': str(row['feature']),
                'importance': float(row['importance'])
            })
        
        # Prepare results
        results = {
            'model_type': 'XGBoost',
            'training_timestamp': datetime.now().isoformat(),
            'hyperparameters': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in params.items()},
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'auc': float(auc)
            },
            'feature_importance_top10': feature_importance_json,
            'training_info': {
                'num_features': int(X_train.shape[1]),
                'num_training_samples': int(len(X_train)),
                'num_validation_samples': int(len(X_val)),
                'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') else int(num_round),
                'label_encoder_used': bool(label_encoder is not None)
            }
        }
        
        # Add label mapping if encoder was used
        if label_encoder is not None:
            # Convert label mapping to JSON-serializable format
            json_label_mapping = {str(k): int(v) for k, v in label_mapping.items()}
            results['label_mapping'] = json_label_mapping
        
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Save model
        logger.info("💾 Saving model...")
        model_path = os.path.join(args.model_dir, 'xgboost-model')
        model.save_model(model_path)
        
        # Save model as pickle 
        joblib.dump(model, os.path.join(args.model_dir, 'XGBoost_model.pkl'))
        
        # Save label encoder if used
        if label_encoder is not None:
            joblib.dump(label_encoder, os.path.join(args.model_dir, 'label_encoder.pkl'))
        
        # Save metrics and results
        with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        # Save feature importance
        feature_importance_df.to_csv(
            os.path.join(args.model_dir, 'XGBoost_feature_importance.csv'), 
            index=False
        )
        
        # Save classification report
        class_report = classification_report(y_val_encoded, y_val_pred, output_dict=True)
        with open(os.path.join(args.model_dir, 'classification_report.json'), 'w') as f:
            json.dump(class_report, f, indent=2, cls=NumpyEncoder)
        
        # Save confusion matrix
        cm = confusion_matrix(y_val_encoded, y_val_pred)
        np.save(os.path.join(args.model_dir, 'confusion_matrix.npy'), cm)
        
        logger.info("✅ Model training completed successfully!")
        logger.info(f"📁 Model saved to: {args.model_dir}")
        
        # Print summary
        print("=" * 50)
        print("XGBOOST TRAINING SUMMARY")
        print("=" * 50)
        print(f"Model Type: XGBoost")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Best Iteration: {model.best_iteration if hasattr(model, 'best_iteration') else num_round}")
        if label_encoder is not None:
            print(f"Label Mapping: {label_mapping}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise e
    
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_params({
            "max_depth": max_depth,
            "eta": eta,
            "num_round": num_round
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc
        })
        mlflow.xgboost.log_model(model, "model")



if __name__ == '__main__':
    main()