import argparse
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import json
import os
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model_artifacts(model_dir, model_type):
    """Load model artifacts from SageMaker training job"""
    
    if model_type.lower() == 'xgboost':
        # Try to load XGBoost model
        xgb_model_path = os.path.join(model_dir, 'xgboost-model')
        if os.path.exists(xgb_model_path):
            model = xgb.Booster()
            model.load_model(xgb_model_path)
            return model, None
        else:
            # Fallback to pickle
            model = joblib.load(os.path.join(model_dir, 'XGBoost_model.pkl'))
            return model, None
            
    elif model_type.lower() == 'logistic_regression':
        model = joblib.load(os.path.join(model_dir, 'LogisticRegression_model.pkl'))
        scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        return model, scaler
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def evaluate_model(model, scaler, X_test, y_test, model_type):
    """Evaluate a single model and return metrics"""
    
    if model_type.lower() == 'xgboost':
        # Check if it's a Booster or sklearn-style model
        if hasattr(model, 'predict_proba'):
            # XGBoost sklearn-style interface
            y_pred_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)
        elif hasattr(model, 'get_booster') or str(type(model)).find('Booster') != -1:
            # XGBoost Booster interface - needs DMatrix
            dtest = xgb.DMatrix(X_test)
            y_pred_prob = model.predict(dtest)
            y_pred = (y_pred_prob > 0.5).astype(int)
        else:
            # Try joblib-loaded XGBoost model
            try:
                # Try sklearn interface first
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            except:
                # Fall back to DMatrix
                dtest = xgb.DMatrix(X_test)
                y_pred_prob = model.predict(dtest)
                y_pred = (y_pred_prob > 0.5).astype(int)
        
    elif model_type.lower() == 'logistic_regression':
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'auc': float(roc_auc_score(y_test, y_pred_prob))
    }
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, class_report, cm, y_pred, y_pred_prob

def create_comparison_plots(xgb_metrics, lr_metrics, xgb_cm, lr_cm, 
                          y_test, xgb_probs, lr_probs, output_dir):
    """Create comparison plots (enhanced from your existing visualization)"""
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Metrics comparison
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    xgb_values = [xgb_metrics[m] for m in metrics_names]
    lr_values = [lr_metrics[m] for m in metrics_names]
    
    x_pos = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 0].bar(x_pos - width/2, xgb_values, width, label='XGBoost', alpha=0.8, color='skyblue')
    axes[0, 0].bar(x_pos + width/2, lr_values, width, label='Logistic Regression', alpha=0.8, color='lightcoral')
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Model Performance Comparison')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(metrics_names, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. XGBoost Confusion Matrix
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('XGBoost Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('Actual')
    
    # 3. Logistic Regression Confusion Matrix
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Reds', ax=axes[0, 2])
    axes[0, 2].set_title('Logistic Regression Confusion Matrix')
    axes[0, 2].set_xlabel('Predicted')
    axes[0, 2].set_ylabel('Actual')
    
    # 4. ROC Curves
    from sklearn.metrics import roc_curve
    
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    
    axes[1, 0].plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_metrics["auc"]:.3f})', color='blue')
    axes[1, 0].plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_metrics["auc"]:.3f})', color='red')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title('ROC Curves Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Prediction Distributions
    axes[1, 1].hist(xgb_probs, bins=30, alpha=0.7, label='XGBoost', color='skyblue')
    axes[1, 1].hist(lr_probs, bins=30, alpha=0.7, label='Logistic Regression', color='lightcoral')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Probability Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Model Agreement
    agreement = (xgb_probs > 0.5) == (lr_probs > 0.5)
    agreement_rate = agreement.mean()
    
    axes[1, 2].pie([agreement_rate, 1-agreement_rate], 
                   labels=['Agreement', 'Disagreement'],
                   autopct='%1.1f%%',
                   colors=['lightgreen', 'salmon'])
    axes[1, 2].set_title(f'Model Agreement Rate: {agreement_rate:.1%}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main evaluation function that compares XGBoost and Logistic Regression"""
    
    parser = argparse.ArgumentParser(description='Model evaluation and comparison')
    
    # Input paths
    parser.add_argument('--xgboost-model', type=str, default='/opt/ml/processing/model/xgboost')
    parser.add_argument('--lr-model', type=str, default='/opt/ml/processing/model/logistic_regression')
    parser.add_argument('--test-data', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--baseline-data', type=str, default='/opt/ml/processing/baseline')
    
    # Output paths
    parser.add_argument('--evaluation-output', type=str, default='/opt/ml/processing/evaluation')
    parser.add_argument('--plots-output', type=str, default='/opt/ml/processing/plots')
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting model evaluation and comparison")
    
    # Create output directories
    os.makedirs(args.evaluation_output, exist_ok=True)
    os.makedirs(args.plots_output, exist_ok=True)
    
    try:
        # FIXED: Load test data without headers
        logger.info("📖 Loading test data...")
        test_files = [f for f in os.listdir(args.test_data) if f.endswith('.csv')]
        if not test_files:
            raise FileNotFoundError(f"No CSV files found in {args.test_data}")
        
        test_df = pd.read_csv(os.path.join(args.test_data, test_files[0]), header=None)
        
        # FIXED: Handle data without headers (SageMaker standard format)
        # First column is target, rest are features
        X_test = test_df.iloc[:, 1:]  # All columns except first
        y_test = test_df.iloc[:, 0]   # First column is target
        
        logger.info(f"📊 Test data shape: {X_test.shape}")
        logger.info(f"🎯 Test target distribution: {y_test.value_counts().to_dict()}")
        
        # Load models
        logger.info("🔄 Loading XGBoost model...")
        xgb_model, _ = load_model_artifacts(args.xgboost_model, 'xgboost')
        
        logger.info("🔄 Loading Logistic Regression model...")
        lr_model, lr_scaler = load_model_artifacts(args.lr_model, 'logistic_regression')
        
        # Evaluate XGBoost
        logger.info("📊 Evaluating XGBoost model...")
        xgb_metrics, xgb_report, xgb_cm, xgb_pred, xgb_probs = evaluate_model(
            xgb_model, None, X_test, y_test, 'xgboost'
        )
        
        # Evaluate Logistic Regression
        logger.info("📊 Evaluating Logistic Regression model...")
        lr_metrics, lr_report, lr_cm, lr_pred, lr_probs = evaluate_model(
            lr_model, lr_scaler, X_test, y_test, 'logistic_regression'
        )
        
        # Compare models and determine best
        logger.info("🏆 Comparing models...")
        
        # Calculate composite scores
        xgb_composite = (xgb_metrics['accuracy'] + xgb_metrics['f1_score'] + xgb_metrics['auc']) / 3
        lr_composite = (lr_metrics['accuracy'] + lr_metrics['f1_score'] + lr_metrics['auc']) / 3
        
        best_model = 'xgboost' if xgb_composite > lr_composite else 'logistic_regression'
        best_metrics = xgb_metrics if best_model == 'xgboost' else lr_metrics
        
        logger.info(f"🏆 Best model: {best_model}")
        logger.info(f"📊 XGBoost composite score: {xgb_composite:.4f}")
        logger.info(f"📊 Logistic Regression composite score: {lr_composite:.4f}")
        
        # Create comparison plots (like your existing visualization)
        logger.info("📈 Creating comparison plots...")
        create_comparison_plots(
            xgb_metrics, lr_metrics, xgb_cm, lr_cm,
            y_test, xgb_probs, lr_probs, args.plots_output
        )
        
        # Prepare evaluation results (similar to your existing model_comparison.csv)
        evaluation_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_samples': int(len(X_test)),
            'models': {
                'xgboost': {
                    'metrics': xgb_metrics,
                    'composite_score': float(xgb_composite),
                    'classification_report': xgb_report
                },
                'logistic_regression': {
                    'metrics': lr_metrics,
                    'composite_score': float(lr_composite),
                    'classification_report': lr_report
                }
            },
            'best_model': {
                'name': best_model,
                'accuracy': float(best_metrics['accuracy']),
                'f1_score': float(best_metrics['f1_score']),
                'auc': float(best_metrics['auc']),
                'composite_score': float(xgb_composite if best_model == 'xgboost' else lr_composite),
                'metrics_path': f"{args.evaluation_output}/best_model_metrics.json"
            },
            'model_comparison': {
                'accuracy_difference': float(abs(xgb_metrics['accuracy'] - lr_metrics['accuracy'])),
                'f1_difference': float(abs(xgb_metrics['f1_score'] - lr_metrics['f1_score'])),
                'auc_difference': float(abs(xgb_metrics['auc'] - lr_metrics['auc'])),
                'agreement_rate': float((xgb_pred == lr_pred).sum() / len(xgb_pred))
            }
        }
        
        # Save evaluation results
        logger.info("💾 Saving evaluation results...")
        
        # Main evaluation file
        with open(os.path.join(args.evaluation_output, 'evaluation.json'), 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Best model metrics (for SageMaker Pipeline property file)
        with open(os.path.join(args.evaluation_output, 'best_model_metrics.json'), 'w') as f:
            json.dump(best_metrics, f, indent=2)
        
        # Model comparison CSV (like your existing output)
        comparison_df = pd.DataFrame({
            'Model': ['XGBoost', 'Logistic Regression'],
            'Accuracy': [xgb_metrics['accuracy'], lr_metrics['accuracy']],
            'Precision': [xgb_metrics['precision'], lr_metrics['precision']],
            'Recall': [xgb_metrics['recall'], lr_metrics['recall']],
            'F1_Score': [xgb_metrics['f1_score'], lr_metrics['f1_score']],
            'AUC': [xgb_metrics['auc'], lr_metrics['auc']],
            'Composite_Score': [xgb_composite, lr_composite]
        })
        
        comparison_df.to_csv(os.path.join(args.evaluation_output, 'model_comparison.csv'), index=False)
        
        # Save confusion matrices
        np.save(os.path.join(args.evaluation_output, 'xgboost_confusion_matrix.npy'), xgb_cm)
        np.save(os.path.join(args.evaluation_output, 'lr_confusion_matrix.npy'), lr_cm)
        
        logger.info("✅ Model evaluation completed successfully!")
        
        # Print summary (like your existing output)
        print("=" * 60)
        print("MODEL EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Test samples: {len(X_test)}")
        print(f"\nXGBoost Performance:")
        print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {xgb_metrics['f1_score']:.4f}")
        print(f"  AUC: {xgb_metrics['auc']:.4f}")
        print(f"  Composite: {xgb_composite:.4f}")
        print(f"\nLogistic Regression Performance:")
        print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
        print(f"  F1-Score: {lr_metrics['f1_score']:.4f}")
        print(f"  AUC: {lr_metrics['auc']:.4f}")
        print(f"  Composite: {lr_composite:.4f}")
        print(f"\n🏆 Best Model: {best_model.upper()}")
        print(f"📊 Model Agreement Rate: {evaluation_results['model_comparison']['agreement_rate']:.1%}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        raise e

if __name__ == '__main__':
    main()