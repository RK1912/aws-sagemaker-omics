import sagemaker
from sagemaker.sklearn.estimator import SKLearn
import boto3

#from dotenv import load_dotenv
#load_dotenv()
import os

role = os.getenv("SAGEMAKER_ROLE")


# --- Configuration ---
bucket = "omics-ml"
script_input_path = f"s3://{bucket}/olink_COVID_19_data_labelled.csv"
output_path = f"s3://{bucket}/sagemaker-output"

# --- SageMaker session ---
sagemaker_session = sagemaker.Session()
region = boto3.Session().region_name

# --- Launch training job ---
sklearn_estimator = SKLearn(
    entry_point="train.py",
    source_dir=".",  # assumes train.py + other files are in this directory
    role=role,
    instance_type="ml.m5.large",  # free tier eligible
    framework_version="1.2-1",     # scikit-learn compatible
    py_version="py3",
    output_path=output_path,
    base_job_name="olink-covid-model",
    hyperparameters={
        "s3_input": script_input_path
    }
)

sklearn_estimator.fit()
