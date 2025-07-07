# Project Overview

This project demonstrates how to build an end-to-end ML pipeline on AWS SageMaker for binary classification of COVID-19 status using [COVID-19 Public OLINK dataset](https://data.mendeley.com/datasets/2cbxgsn7vx/1?utm_source=chatgpt.com) The pipeline compares XGBoost and Logistic Regression models, automatically selects the best performer, and registers it for deployment. This ML pipeline is developed and tested on SageMaker Studio Lab, and deployed on AWS Sagemaker ( Free Tier Account ) using AWS Codebuild. 

## Key Features
🔄 Automated MLOps Pipeline: Complete SageMaker Pipelines implementation  
⚡ Parallel Model Training: XGBoost and Logistic Regression trained simultaneously  
📊 Comprehensive Evaluation: Automated model comparison with visualizations  
🎯 Conditional Registration: Automatic best model selection and registration  
💰 Cost-Optimized: Free Tier compatible with smart resource management  
🔍 Production-Ready: Robust error handling and comprehensive logging

## Project Structure

```
aws-sagemaker-omics/
├── README.md
├── requirements.txt
├── config/
│   └── free_tier_config.json          # Pipeline configuration
├── src/
│   ├── preprocessing/
│   │   └── olink_preprocessor.py       # Data preprocessing logic
│   ├── training/
│   │   ├── train_xgboost.py           # XGBoost training script
│   │   ├── train_logistic_regression.py  # Logistic Regression training
│   │   └── model_evaluation.py        # Model comparison & evaluation
│   └── pipeline/
│       ├── pipeline_definition.py     # Main pipeline orchestration
│       └── deploy_pipeline.py         # Pipeline deployment script
└── data/
    └── olink_COVID_19_data_labelled.csv  # Input dataset
```

## Getting Started
### Prerequisites

- AWS Account with SageMaker, S3, Codebuild, ECR
- Python 3.8+
- AWS CLI configured
- SageMaker execution role with appropriate permissions

### Installation

- Clone the repository
  ```
  git clone https://github.com/RK1912/aws-sagemaker-omics.git
  cd aws-sagemaker-omics
  ```

- Install dependencies
  ```
  pip install -r requirements.txt
  ```

- Configure AWS credentials
  ```
  aws configure
  ```

- Set up environment variables
  ```
  export SAGEMAKER_ROLE="arn:aws:iam::YOUR-ACCOUNT:role/SageMakerExecutionRole"
  export SAGEMAKER_IMAGE_URI="YOUR-ECR-IMAGE-URI" 
  ```

- Update configuration
  Edit config/free_tier_config.json with your AWS details:
  ```
  json{
    "aws": {
      "region": "your-region",
      "bucket": "your-s3-bucket",
      "sagemaker_role": "SAGEMAKER_ROLE"
    },
    "pipeline": {
      "name": "olink-covid-classification"
    },
    "instances": {
      "processing": {"type": "ml.t3.medium"},
      "training": {"type": "ml.m5.large"}
    }
  }
  ```

### Quick Start

1. Upload your data to S3
   ```
   aws s3 cp data/olink_COVID_19_data_labelled.csv s3://your-bucket/
   ```

2. Trigger the pipeline on Sagemaker via Sagemaker studio lab or local
   ```
   python3 scripts/trigger.py
   ```

4. Monitor progress
   Go to AWS Console → SageMaker → Pipelines
   Cloudwatch logs : /aws/sagemaker/ProcessingJobs and /aws/sagemaker/TrainingJobs

### Pipeline Steps

1. Data Preprocessing (PreprocessOLINKData)
   - Data validation and cleaning
   - Train/validation/test splits (70/10/20)
   - Feature scaling and engineering
   - Outlier detection
   - PCA transformation (optional)

2. Parallel Model Training
   - XGBoost Training (TrainXGBoostModel)
     - Hyperparameter: max_depth, eta, num_rounds
     - Early stopping to prevent overfitting
     - Built-in cross-validation
   - Logistic Regression Training (TrainLogisticRegressionModel)
     - Hyperparameters: C, penalty, solver
     - Feature scaling with StandardScaler
3. Model Evaluation (EvaluateModels)
   - Comprehensive metrics: Accuracy, F1, AUC, Precision, Recall
   - Confusion matrices and ROC curves
   - Automated best model selection
4. Conditional Registration (ConditionalModelRegistration)
   - Register model only if accuracy ≥ threshold (default: 0.85)
   - Model versioning and metadata tracking

### S3 Output Structure
s3://your-bucket/pipeline-output/
```
├── processed/
│   ├── train/           # Training data
│   ├── validation/      # Validation data
│   ├── test/           # Test data
│   └── baseline/       # Baseline statistics
├── models/
│   ├── xgboost/        # XGBoost model artifacts
│   └── logistic_regression/  # LR model artifacts
├── evaluation/
│   ├── evaluation.json      # Detailed metrics
│   ├── model_comparison.csv # Side-by-side comparison
│   └── best_model_metrics.json  # Best model details
└── plots/
    └── model_comparison.png    # Visualization plots
```

## Future work 
1. Integration with AWS Glue for ETL preprocessing of data
2. Integration with RDS and/or DynamoDB for storing raw data and metadata
