import argparse
import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Preprocessor script with Exploratory data analysis, outlier detection and splitting data into train and test"""
    
    parser = argparse.ArgumentParser(description='OLINK data preprocessing')
    
    # SageMaker arguments
    parser.add_argument('--input-data', type=str, default='/opt/ml/processing/input')
    parser.add_argument('--train-data', type=str, default='/opt/ml/processing/train')
    parser.add_argument('--validation-data', type=str, default='/opt/ml/processing/validation')
    parser.add_argument('--test-data', type=str, default='/opt/ml/processing/test')
    parser.add_argument('--baseline-data', type=str, default='/opt/ml/processing/baseline')
    parser.add_argument('--plots-output', type=str, default='/opt/ml/processing/baseline')
    
    # Processing arguments
    parser.add_argument('--test-split-ratio', type=float, default=0.2)
    parser.add_argument('--validation-split-ratio', type=float, default=0.1)
    parser.add_argument('--apply-pca', type=str, default='true')
    parser.add_argument('--detect-outliers', type=str, default='true')
    parser.add_argument('--pca-components', type=int, default=10)
    parser.add_argument('--outlier-contamination', type=float, default=0.1)
    parser.add_argument('--random-state', type=int, default=42)
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting OLINK data preprocessing")
    
    # Create output directories
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.validation_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)
    os.makedirs(args.baseline_data, exist_ok=True)
    os.makedirs(args.plots_output, exist_ok=True)
    
    subprocess.check_call(["sudo", "chown", "-R", "sagemaker-user", "/opt/ml/processing/"])
    
    
    
    # Initialize variables
    outliers = None  # Initialize to avoid undefined variable error
    
    try:
        # Load data 
        logger.info("📖 Loading OLINK dataset...")
        data_files = os.listdir(args.input_data)
        csv_files = [f for f in data_files if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV files found in input directory")
        
        # Load the first CSV file
        data_path = os.path.join(args.input_data, csv_files[0])
        df = pd.read_csv(data_path)
        
        logger.info(f"📊 Dataset shape: {df.shape}")
        logger.info(f"📋 Columns: {list(df.columns)}")
        
        # Basic data info
        logger.info("🔍 Dataset Info:")
        logger.info(f"  - Missing values: {df.isnull().sum().sum()}")
        logger.info(f"  - Duplicate rows: {df.duplicated().sum()}")
        
        # Identify target column
        target_candidates = ['target', 'label', 'class', 'COVID19', 'diagnosis', 'COVID']
        target_column = None
        
        for col in target_candidates:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            # If no obvious target column, assume last column is target
            target_column = df.columns[-1]
            logger.warning(f"⚠️ No obvious target column found, using '{target_column}'")
        
        logger.info(f"🎯 Target column: {target_column}")
        logger.info(f"🎯 Target distribution: {df[target_column].value_counts().to_dict()}")
        
        # Separate features and target
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Handle missing values
        if X.isnull().sum().sum() > 0:
            logger.info("🔧 Handling missing values...")
            X = X.fillna(X.median())
        
        # Remove duplicate rows
        if df.duplicated().sum() > 0:
            logger.info("🔧 Removing duplicate rows...")
            df_clean = df.drop_duplicates()
            X = df_clean.drop(target_column, axis=1)
            y = df_clean[target_column]
        
        # Feature selection (remove constant or near-constant features)
        logger.info("🔧 Feature selection...")
        variance_threshold = 0.01
        feature_variances = X.var()
        low_variance_features = feature_variances[feature_variances < variance_threshold].index
        
        if len(low_variance_features) > 0:
            logger.info(f"🗑️ Removing {len(low_variance_features)} low variance features")
            X = X.drop(low_variance_features, axis=1)
        
        logger.info(f"📊 Final feature count: {X.shape[1]}")
        
        # Outlier detection 
        outliers = pd.Series([False] * len(X), index=X.index)  # Initialize outliers
        
        if args.detect_outliers.lower() == 'true':
            logger.info("🔍 Detecting outliers...")
            
            # Standardize features for outlier detection
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Isolation Forest for outlier detection
            iso_forest = IsolationForest(
                contamination=args.outlier_contamination,
                random_state=args.random_state
            )
            outlier_labels = iso_forest.fit_predict(X_scaled)
            outliers = pd.Series(outlier_labels == -1, index=X.index)
            
            # Create outlier plots 
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Outlier scores
            plt.subplot(1, 2, 1)
            outlier_scores = iso_forest.score_samples(X_scaled)
            plt.hist(outlier_scores, bins=50, alpha=0.7, color='skyblue')
            plt.xlabel('Outlier Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Outlier Scores')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Outliers visualization
            plt.subplot(1, 2, 2)
            plt.scatter(range(len(outlier_scores)), outlier_scores, 
                       c=['red' if x else 'blue' for x in outliers], alpha=0.6)
            plt.xlabel('Sample Index')
            plt.ylabel('Outlier Score')
            plt.title(f'Outlier Detection (Found {outliers.sum()} outliers)')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            outlier_plot_file = "/opt/ml/processing/baseline/outlier_detection.png"
            plt.savefig(outlier_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"🔍 Found {outliers.sum()} outliers ({outliers.sum()/len(X)*100:.1f}%)")
            
            # Option to remove outliers
            # X = X[~outliers]
            # y = y[~outliers]
            # logger.info(f"📊 Data shape after outlier removal: {X.shape}")
        
        # PCA Analysis 
        if args.apply_pca.lower() == 'true':
            logger.info("🔬 Applying PCA analysis...")
            
            # Standardize features for PCA
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA
            pca = PCA(n_components=min(args.pca_components, X.shape[1]))
            X_pca = pca.fit_transform(X_scaled)
            
            # Create PCA plots 
            plt.figure(figsize=(15, 5))
            
            # Plot 1: Explained variance
            plt.subplot(1, 3, 1)
            plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    np.cumsum(pca.explained_variance_ratio_), 'bo-')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance Ratio')
            plt.title('PCA: Cumulative Explained Variance')
            plt.grid(True, alpha=0.3)
            
            # Plot 2: First two components scatter
            plt.subplot(1, 3, 2)
            colors = ['red' if label == 1 else 'blue' for label in y]
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.6)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.title('PCA: First Two Components')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Target distribution
            plt.subplot(1, 3, 3)
            y.value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.title('Class Distribution')
            plt.xticks(rotation=0)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            pca_plot_file = "/opt/ml/processing/baseline/pca_analysis.png"
            plt.savefig(pca_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"🔬 PCA completed. First {args.pca_components} components explain {pca.explained_variance_ratio_[:args.pca_components].sum():.2%} of variance")
        
        # Data splitting
        logger.info("🔄 Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=args.test_split_ratio,
            random_state=args.random_state,
            stratify=y
        )
        
        # Second split: train vs validation
        val_size = args.validation_split_ratio / (1 - args.test_split_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=args.random_state,
            stratify=y_temp
        )
        
        logger.info(f"📊 Data split:")
        logger.info(f"  - Training: {X_train.shape[0]} samples")
        logger.info(f"  - Validation: {X_val.shape[0]} samples") 
        logger.info(f"  - Test: {X_test.shape[0]} samples")
        
        # Save data in SageMaker format (target first, no headers for training compatibility)
        logger.info("💾 Saving processed datasets...")
        
        # Prepare datasets with target column first (SageMaker standard requirement)
        train_final = pd.concat([y_train, X_train], axis=1)
        val_final = pd.concat([y_val, X_val], axis=1)
        test_final = pd.concat([y_test, X_test], axis=1)
        
        # Save without headers for training compatibility
        train_final.to_csv(os.path.join(args.train_data, 'train.csv'), index=False, header=False)
        val_final.to_csv(os.path.join(args.validation_data, 'validation.csv'), index=False, header=False)
        test_final.to_csv(os.path.join(args.test_data, 'test.csv'), index=False, header=False)
        
        # Save baseline data for model monitoring (with headers for evaluation script)
        baseline_sample = train_final.sample(n=min(1000, len(train_final)), random_state=args.random_state)
        baseline_sample.to_csv(os.path.join(args.baseline_data, 'baseline.csv'), index=False, header=True)
        
        # Save preprocessing metadata
        preprocessing_info = {
            'preprocessing_timestamp': datetime.now().isoformat(),
            'original_shape': list(df.shape),
            'final_shape': list(X.shape),
            'target_column': target_column,
            'features_removed': len(low_variance_features),
            'outliers_detected': int(outliers.sum()) if args.detect_outliers.lower() == 'true' else 0,
            'train_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test),
            'target_distribution': y.value_counts().to_dict(),
            'feature_names': X.columns.tolist(),
            'preprocessing_parameters': {
                'test_split_ratio': args.test_split_ratio,
                'validation_split_ratio': args.validation_split_ratio,
                'apply_pca': args.apply_pca,
                'detect_outliers': args.detect_outliers,
                'pca_components': args.pca_components,
                'outlier_contamination': args.outlier_contamination
            }
        }
        
        with open(os.path.join(args.baseline_data, 'preprocessing_info.json'), 'w') as f:
            json.dump(preprocessing_info, f, indent=2)
        
        # Save feature names for reference
        feature_info = {
            'target_column': target_column,
            'feature_names': X.columns.tolist(),
            'n_features': len(X.columns)
        }
        
        with open(os.path.join(args.baseline_data, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info("✅ Preprocessing completed successfully!")
        
        # Print summary
        print("=" * 50)
        print("OLINK DATA PREPROCESSING SUMMARY")
        print("=" * 50)
        print(f"Original dataset: {df.shape[0]} samples, {df.shape[1]} features")
        print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target column: {target_column}")
        print(f"Class distribution: {dict(y.value_counts())}")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        if args.detect_outliers.lower() == 'true':
            print(f"Outliers detected: {outliers.sum()}")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        raise e

if __name__ == '__main__':
    main()