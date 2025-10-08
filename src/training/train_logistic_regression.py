#!/usr/bin/env python3

import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import os
import json
import logging
import sys
from datetime import datetime
import mlflow
import mlflow.sklearn

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Logistic Regression training"""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Logistic Regression training for OLINK data')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output'))
    
    # Hyperparameters with defaults
    parser.add_argument('-C', '--C', type=str, default='1.0', dest='C')
    parser.add_argument('--max_iter', type=str, default='1000')
    parser.add_argument('--penalty', type=str, default='l2')
    parser.add_argument('--solver', type=str, default='liblinear')
    parser.add_argument('--random_state', type=str, default='42')
    parser.add_argument('--class_weight', type=str, default='balanced')
    
    # Parse arguments
    try:
        args = parser.parse_args()
        logger.info("✅ Arguments parsed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to parse arguments: {e}")
        sys.exit(1)
    
    # Convert string hyperparameters to appropriate types
    try:
        c_value = float(args.C)
        max_iter_value = int(args.max_iter)
        penalty_value = str(args.penalty)
        solver_value = str(args.solver)
        random_state_value = int(args.random_state)
        class_weight_value = str(args.class_weight)
        
        logger.info("✅ Hyperparameters converted successfully")
    except Exception as e:
        logger.error(f"❌ Failed to convert hyperparameters: {e}")
        sys.exit(1)
    
    logger.info("🚀 Starting Logistic Regression training for OLINK COVID-19 classification")
    logger.info(f"📁 Model directory: {args.model_dir}")
    logger.info(f"📊 Training data: {args.train}")
    logger.info(f"📈 Validation data: {args.validation}")
    logger.info(f"⚙️ Hyperparameters: C={c_value}, max_iter={max_iter_value}, penalty={penalty_value}, solver={solver_value}")
    
    try:
        # Verify directories exist
        if not os.path.exists(args.train):
            raise FileNotFoundError(f"Training directory not found: {args.train}")
        if not os.path.exists(args.validation):
            raise FileNotFoundError(f"Validation directory not found: {args.validation}")
        
        # Load training and validation data
        logger.info("📖 Loading training data...")
        train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
        if not train_files:
            raise FileNotFoundError(f"No CSV files found in {args.train}")
        
        train_df = pd.read_csv(os.path.join(args.train, train_files[0]), header=None)
        logger.info(f"📊 Training data loaded: {train_df.shape}")
        
        logger.info("📖 Loading validation data...")
        val_files = [f for f in os.listdir(args.validation) if f.endswith('.csv')]
        if not val_files:
            raise FileNotFoundError(f"No CSV files found in {args.validation}")
            
        val_df = pd.read_csv(os.path.join(args.validation, val_files[0]), header=None)
        logger.info(f"📈 Validation data loaded: {val_df.shape}")
        
        # Handle data without headers (SageMaker standard format)
        # First column is target, rest are features
        X_train = train_df.iloc[:, 1:]  # All columns except first
        y_train = train_df.iloc[:, 0]   # First column is target
        X_val = val_df.iloc[:, 1:]
        y_val = val_df.iloc[:, 0]
        
        logger.info(f"🎯 Target distribution in training: {y_train.value_counts().to_dict()}")
        logger.info(f"🎯 Target distribution in validation: {y_val.value_counts().to_dict()}")
        
        # Handle string labels by converting to numeric
        label_encoder = None
        label_mapping = None
        
        if y_train.dtype == 'object' or y_train.dtype.name == 'string':
            logger.info("🔄 Converting string labels to numeric...")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_val_encoded = label_encoder.transform(y_val)
            
            # Create label mapping
            label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
            logger.info(f"🔢 Label mapping: {label_mapping}")
        else:
            # Ensure labels are numeric
            y_train_encoded = pd.to_numeric(y_train, errors='coerce')
            y_val_encoded = pd.to_numeric(y_val, errors='coerce')
            
            # Check for any conversion issues
            if y_train_encoded.isna().any() or y_val_encoded.isna().any():
                raise ValueError("Unable to convert labels to numeric. Please check your data.")
        
        
        logger.info("🔄 Ensuring features are numeric...")
        X_train = X_train.apply(pd.to_numeric, errors='coerce')
        X_val = X_val.apply(pd.to_numeric, errors='coerce')
        
        # Handle missing values
        if X_train.isna().any().any():
            logger.warning("⚠️ Found NaN values in training features after conversion. Filling with 0.")
            X_train = X_train.fillna(0)
        
        if X_val.isna().any().any():
            logger.warning("⚠️ Found NaN values in validation features after conversion. Filling with 0.")
            X_val = X_val.fillna(0)
        
        # Feature scaling 
        logger.info("🔄 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Handle class_weight parameter
        if class_weight_value.lower() == 'none':
            class_weight = None
        elif class_weight_value.lower() == 'balanced':
            class_weight = 'balanced'
        else:
            class_weight = class_weight_value
        
        # Train Logistic Regression model
        logger.info("🏋️ Starting Logistic Regression training...")
        model = LogisticRegression(
            C=c_value,
            max_iter=max_iter_value,
            penalty=penalty_value,
            solver=solver_value,
            random_state=random_state_value,
            class_weight=class_weight,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        logger.info("✅ Training completed!")
        
        # Make predictions
        logger.info("🔮 Making predictions on validation set...")
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics using encoded labels
        accuracy = accuracy_score(y_val_encoded, y_val_pred)
        f1 = f1_score(y_val_encoded, y_val_pred)
        auc = roc_auc_score(y_val_encoded, y_val_pred_prob)
        
        logger.info(f"📊 Validation Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Validation F1-Score: {f1:.4f}")
        logger.info(f"📊 Validation AUC: {auc:.4f}")
        
        # Feature importance (coefficients) with numeric feature names
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        coefficients = model.coef_[0]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Convert feature importance to JSON-serializable format
        feature_importance_json = []
        for _, row in feature_importance_df.head(10).iterrows():
            feature_importance_json.append({
                'feature': str(row['feature']),
                'coefficient': float(row['coefficient']),
                'abs_coefficient': float(row['abs_coefficient'])
            })
        
        # Prepare results 
        results = {
            'model_type': 'LogisticRegression',
            'training_timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'C': float(c_value),
                'max_iter': int(max_iter_value),
                'penalty': str(penalty_value),
                'solver': str(solver_value),
                'class_weight': str(class_weight_value)
            },
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
                'convergence_achieved': bool(model.n_iter_[0] < max_iter_value),
                'iterations_used': int(model.n_iter_[0]),
                'label_encoder_used': bool(label_encoder is not None)
            }
        }
        
        # label mapping
        if label_encoder is not None and label_mapping is not None:
            # Convert label mapping to JSON-serializable format
            json_label_mapping = {str(k): int(v) for k, v in label_mapping.items()}
            results['label_mapping'] = json_label_mapping
        
        # Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Save model and scaler
        logger.info("💾 Saving model and scaler...")
        joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
        joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.joblib'))
        
        # Save label encoder 
        if label_encoder is not None:
            joblib.dump(label_encoder, os.path.join(args.model_dir, 'label_encoder.joblib'))
        
        # Save metrics and results
        with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2, cls=NumpyEncoder)
        
        # Save feature importance
        feature_importance_df.to_csv(
            os.path.join(args.model_dir, 'feature_importance.csv'),
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
        
        # Print summary for SageMaker logs
        print("=" * 50)
        print("LOGISTIC REGRESSION TRAINING SUMMARY")
        print("=" * 50)
        print(f"Model Type: Logistic Regression")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Convergence: {'Yes' if model.n_iter_[0] < max_iter_value else 'No'}")
        if label_encoder is not None and label_mapping is not None:
            print(f"Label Mapping: {label_mapping}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        # Print to stdout for SageMaker to capture
        print(f"ERROR: {str(e)}")
        print(f"TRACEBACK: {traceback.format_exc()}")
        sys.exit(1)

    with mlflow.start_run():
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_params({
            "C": c_value,
            "max_iter": max_iter_value,
            "penalty": penalty_value,
            "solver": solver_value,
            "class_weight": class_weight_value
        })
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1,
            "auc": auc
        })
        mlflow.sklearn.log_model(model, "model")

    

if __name__ == '__main__':
    main()