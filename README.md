# aws-sagemaker-omics

This project was developed to learn about AWS services that can be used to deploy a full ML workflow. This ML pipeline is developed and tested on SageMaker Studio Lab, and deployed on AWS Sagemaker ( Free Tier Account ) using AWS Codebuild. 
The pipeline uses the [COVID-19 Public OLINK dataset](https://data.mendeley.com/datasets/2cbxgsn7vx/1?utm_source=chatgpt.com) to classify patients into COVID-19 or Healthy categories using Logistic regression and Xgboost classifiers.


## Components 
### buildspec.yml 
This YAML file is used by AWS Codebuild to define and build the deployment steps such as installing dependencies, run scripts, etc. 

### requirements.txt 
Includes all python modules and their versions to be installed. 

### eda.py 
Script to perform Exploratory Data Analysis such as PCA, outlier detection and class label distribution. 

### train.py 
Script to train machine learning models , specifically Xgboost and logistic regression to perform a classification task. 

## Results 
Results are saved automatically to AWS S3 bucket. Results contain the following files:  
  .
```
├── results/
│   ├── LogisticRegression_model.pkl
│   ├── LogisticRegression_feature_importance.csv
│   ├── XGBoost_model.pkl
│   ├── XGBoost_feature_importance.csv
│   ├── model_comparison.csv
    ├── plots/
    │   ├── label_distribution.png
    │   ├── outlier_scores.png
    │   ├── pca_scatter.png
    │   └── outliers_isolation_forests.png

```

