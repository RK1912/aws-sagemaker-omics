import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Enhanced Logistic Regression training for OLINK COVID-19 classification"""
    
    parser = argparse.ArgumentParser(description='Logistic Regression training for OLINK data')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Hyperparameters
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--penalty', type=str, default='l2')
    parser.add_argument('--solver', type=str, default='liblinear')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--class_weight', type=str, default='balanced')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Logistic Regression training for OLINK COVID-19 classification")
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
        
        # Feature scaling (important for logistic regression)
        logger.info("🔄 Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Handle class_weight parameter
        class_weight = None if args.class_weight == 'None' else args.class_weight
        
        # Train Logistic Regression model
        logger.info("🏋️ Starting Logistic Regression training...")
        model = LogisticRegression(
            C=args.C,
            max_iter=args.max_iter,
            penalty=args.penalty,
            solver=args.solver,
            random_state=args.random_state,
            class_weight=class_weight,
            n_jobs=-1  # Use all available cores
        )
        
        model.fit(X_train_scaled, y_train)
        
        logger.info("✅ Training completed!")
        
        # Make predictions
        logger.info("🔮 Making predictions on validation set...")
        y_val_pred = model.predict(X_val_scaled)
        y_val_pred_prob = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_pred_prob)
        
        logger.info(f"📊 Validation Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Validation F1-Score: {f1:.4f}")
        logger.info(f"📊 Validation AUC: {auc:.4f}")
        
        # FIXED: Feature importance (coefficients) with numeric feature names
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        coefficients = model.coef_[0]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Prepare results
        # Prepare results (FIXED: Convert numpy types to Python types)
        results = {
            'model_type': 'LogisticRegression',
            'training_timestamp': datetime.now().isoformat(),
            'hyperparameters': {
                'C': float(args.C),
                'max_iter': int(args.max_iter),
                'penalty': str(args.penalty),
                'solver': str(args.solver),
                'class_weight': str(args.class_weight)
            },
            'metrics': {
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'auc': float(auc)
            },
            'feature_importance_top10': feature_importance_df.head(10).to_dict('records'),
            'training_info': {
                'num_features': int(X_train.shape[1]),
                'num_training_samples': int(len(X_train)),
                'num_validation_samples': int(len(X_val)),
                'convergence_achieved': bool(model.n_iter_[0] < args.max_iter),  # FIXED: Convert to Python bool
                'iterations_used': int(model.n_iter_[0])
            }
        }
        # FIXED: Ensure model directory exists
        os.makedirs(args.model_dir, exist_ok=True)
        
        # Save model and scaler
        logger.info("💾 Saving model and scaler...")
        joblib.dump(model, os.path.join(args.model_dir, 'LogisticRegression_model.pkl'))
        joblib.dump(scaler, os.path.join(args.model_dir, 'scaler.pkl'))
        
        # Save metrics and results
        with open(os.path.join(args.model_dir, 'metrics.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save feature importance (like your existing output)
        feature_importance_df.to_csv(
            os.path.join(args.model_dir, 'LogisticRegression_feature_importance.csv'),
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
        
        # Print summary
        print("=" * 50)
        print("LOGISTIC REGRESSION TRAINING SUMMARY")
        print("=" * 50)
        print(f"Model Type: Logistic Regression")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Convergence: {'Yes' if model.n_iter_[0] < args.max_iter else 'No'}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        raise e

if __name__ == '__main__':
    main()