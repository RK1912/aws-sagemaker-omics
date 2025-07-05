import boto3
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import (
    ProcessingStep, 
    TrainingStep, 
    CreateModelStep,
    TransformStep
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.model import Model
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor  # Add ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.properties import PropertyFile
from sagemaker.inputs import TrainingInput
import json
import os

class OLINKPipeline:
    """
    Enhanced SageMaker Pipeline for OLINK COVID-19 Classification
    Optimized for AWS Free Tier with cost controls
    """
    
    def __init__(self, config_path: str = "config/free_tier_config.json"):
        self.config = self._load_config(config_path)
        self.region = self.config['aws']['region']
        self.bucket = self.config['aws']['bucket']
        self.pipeline_name = self.config['pipeline']['name']

        print(f"🔍 Config sagemaker_role: {self.config['aws']['sagemaker_role']}")
        print(f"🔍 Environment SAGEMAKER_ROLE: {os.getenv('SAGEMAKER_ROLE')}")

        self.role = os.getenv(self.config['aws']['sagemaker_role'])
        
        # DEBUG
        print(f"🔍 Final role: {self.role}")
    
        if not self.role:
            raise ValueError(f"❌ No role found! Looking for env var: {self.config['aws']['sagemaker_role']}")
            
        # Initialize SageMaker session
        self.pipeline_session = PipelineSession()
        self.sagemaker_session = sagemaker.Session()
        
        # Define pipeline parameters
        self._define_parameters()
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _define_parameters(self):
        """Define pipeline parameters for flexibility"""
        
        # Data parameters
        self.input_data_uri = ParameterString(
            name="InputDataUri",
            default_value=f"s3://{self.bucket}/olink_COVID_19_data_labelled.csv"
        )
        
        
        self.instance_count = ParameterInteger(name="InstanceCount", default_value=1)
        
        self.output_prefix = ParameterString(
            name="OutputPrefix",
            default_value=f"s3://{self.bucket}/pipeline-output"
        )
        
        # Model approval parameters
        self.model_approval_status = ParameterString(
            name="ModelApprovalStatus",
            default_value="PendingManualApproval"
        )
        
        self.accuracy_threshold = ParameterFloat(
            name="AccuracyThreshold",
            default_value=0.85
        )
        
        # Instance parameters (Free Tier optimized)
        self.processing_instance_type = ParameterString(
            name="ProcessingInstanceType",
            default_value="ml.t3.medium"  
        )
        
        self.training_instance_type = ParameterString(
            name="TrainingInstanceType", 
            default_value="ml.t3.medium"  
        )
        
        
        # XGBoost hyperparameters
        self.xgb_max_depth = ParameterInteger(
            name="XGBMaxDepth",
            default_value=6
        )
        
        self.xgb_eta = ParameterFloat(
            name="XGBEta", 
            default_value=0.3
        )
        
        self.xgb_num_round = ParameterInteger(
            name="XGBNumRound",
            default_value=100
        )
        
        # Logistic Regression hyperparameters
        self.lr_c = ParameterFloat(
            name="LRC",
            default_value=1.0
        )
        
        self.lr_max_iter = ParameterInteger(
            name="LRMaxIter",
            default_value=1000
        )
    
    def create_preprocessing_step(self):
        """
        Create data preprocessing step
        """
        
        # SKLearn processor for preprocessing (Free Tier optimized)
        script_processor = ScriptProcessor(
            image_uri=os.getenv('SAGEMAKER_IMAGE_URI', 'public.ecr.aws/sagemaker/sagemaker-distribution:3.1-cpu'),
            command=["python3"],
            instance_type=self.processing_instance_type,
            instance_count=self.instance_count,
            base_job_name="olink-preprocessing",
            role=self.role,
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=3600
        )
        
        preprocessing_step = ProcessingStep(
            name="PreprocessOLINKData",
            processor=script_processor,
            inputs=[
                ProcessingInput(
                    source=self.input_data_uri,
                    destination="/opt/ml/processing/input",
                    input_name="raw_data"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="train_data",
                    source="/opt/ml/processing/train",
                    destination = Join(on="/", values=[self.output_prefix, "processed", "train"])
                    
                ),
                ProcessingOutput(
                    output_name="validation_data",
                    source="/opt/ml/processing/validation", 
                    destination=Join(on="/", values=[self.output_prefix,"processed","validation"])
                ),
                ProcessingOutput(
                    output_name="test_data",
                    source="/opt/ml/processing/test",
                    destination=Join(on="/",values=[self.output_prefix,"processed","test"])
                ),
                ProcessingOutput(
                    output_name="baseline_data",
                    source="/opt/ml/processing/baseline",
                    destination=Join(on="/",values=[self.output_prefix,"processed","baseline"])
                )
            ],
            code="src/preprocessing/olink_preprocessor.py",
            job_arguments=[
                "--test-split-ratio", "0.2",
                "--validation-split-ratio", "0.1",
                "--apply-pca", "true",
                "--detect-outliers", "true"
            ]
        )
        
        return preprocessing_step
    
    def create_xgboost_training_step(self, preprocessing_step):
        """
        Create XGBoost training step with your existing logic enhanced
        """
        
        # XGBoost estimator using built-in container
        xgb_estimator = XGBoost(
            entry_point="src/training/train_xgboost.py",
            framework_version="1.0-1",
            instance_type=self.training_instance_type,
            instance_count=1,
            role=self.role,
            base_job_name="olink-xgboost-training",
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=3600,  # 1 hour limit
            use_spot_instances=True,  # Cost optimization
            max_wait_time_in_seconds=7200,  # 2 hour max wait
            hyperparameters={
                "max_depth": self.xgb_max_depth,
                "eta": self.xgb_eta,
                "objective": "binary:logistic", 
                "eval_metric": "auc",
                "num_round": self.xgb_num_round,
                "early_stopping_rounds": 10  # Prevent overfitting & save costs
            }
        )
        
        xgb_training_step = TrainingStep(
            name="TrainXGBoostModel",
            estimator=xgb_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                    content_type="text/csv"
                ),
                "validation": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )
        
        return xgb_training_step
    
    def create_logistic_regression_training_step(self, preprocessing_step):
        """
        Create Logistic Regression training step 
        """
        
        # SKLearn estimator for Logistic Regression
        lr_estimator = SKLearn(
            entry_point="src/training/train_logistic_regression.py",
            framework_version="1.0-1", 
            py_version="py3",
            instance_type=self.training_instance_type,
            role=self.role,
            base_job_name="olink-lr-training",
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=3600,  # 1 hour limit
            use_spot_instances=True,  # Cost optimization
            max_wait_time_in_seconds=7200,
            hyperparameters={
                "C": self.lr_c,
                "max_iter": self.lr_max_iter,
                "penalty": "l2",
                "solver": "liblinear",
                "random_state": 42
            }
        )
        
        lr_training_step = TrainingStep(
            name="TrainLogisticRegressionModel",
            estimator=lr_estimator,
            inputs={
                "train": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri,
                    content_type="text/csv"
                ),
                "validation": TrainingInput(
                    s3_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs["validation_data"].S3Output.S3Uri,
                    content_type="text/csv"
                )
            }
        )
        
        return lr_training_step
    
    def create_evaluation_step(self, preprocessing_step, xgb_step, lr_step):
        """
        Create model evaluation step that compares both models
        Enhanced from your existing comparison logic
        """
        
        # Evaluation processor
        eval_processor = ScriptProcessor(
            image_uri=os.getenv('SAGEMAKER_IMAGE_URI', 'public.ecr.aws/sagemaker/sagemaker-distribution:3.1-cpu'),
            command=["python3"],
            instance_type=self.processing_instance_type,
            instance_count=self.instance_count,
            base_job_name="olink-model-evaluation",
            role=self.role,
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=1800
        )
        
        # Property file for evaluation results
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation", 
            path="evaluation.json"
        )
        
        evaluation_step = ProcessingStep(
            name="EvaluateModels",
            processor=eval_processor,
            inputs=[
                ProcessingInput(
                    source=xgb_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model/xgboost",
                    input_name="xgboost_model"
                ),
                ProcessingInput(
                    source=lr_step.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/model/logistic_regression", 
                    input_name="lr_model"
                ),
                ProcessingInput(
                    source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test",
                    input_name="test_data"
                ),
                ProcessingInput(
                    source=preprocessing_step.properties.ProcessingOutputConfig.Outputs["baseline_data"].S3Output.S3Uri,
                    destination="/opt/ml/processing/baseline",
                    input_name="baseline_data"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source="/opt/ml/processing/evaluation",
                    destination=Join(on="/",values=[self.output_prefix,"evaluation"])
                ),
                ProcessingOutput(
                    output_name="plots",
                    source="/opt/ml/processing/plots", 
                    destination=Join(on="/",values=[self.output_prefix,"plots"])
                )
            ],
            code="src/training/model_evaluation.py",
            property_files=[evaluation_report]
        )
        
        return evaluation_step, evaluation_report
    
    def create_model_registration_step(self, best_training_step, evaluation_step, evaluation_report):
        """
        Create model registration step for the best performing model
        """
        
        # Model metrics for registration
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=evaluation_report,
                    json_path="best_model.metrics_path"
                ),
                content_type="application/json"
            )
        )
        
        ## create model object 
        model = Model(
            image_uri=best_training_step.estimator.training_image_uri(),
            model_data=best_training_step.properties.ModelArtifacts.S3ModelArtifacts,
            sagemaker_session=self.pipeline_session,
            role=self.role
        )
        
        
        # Register best model
        register_model_step = ModelStep(
            name="RegisterBestModel",
            step_args = model.register(
                content_types=["text/csv"],
                response_types=["text/csv"], 
                inference_instances=["ml.m4.xlarge", "ml.m5.xlarge"],  # Free Tier eligible
                transform_instances=["ml.m4.xlarge"],
                model_package_group_name="olink-classification-models",
                approval_status=self.model_approval_status,
                model_metrics=model_metrics,
                description="Best performing model for OLINK COVID-19 classification"
            )
        )
        
        return register_model_step
    
    def create_conditional_registration_step(self, evaluation_step, evaluation_report, 
                                           xgb_step, lr_step):
        """
        Create conditional model registration based on performance threshold
        """
        
        # Condition: Register model only if accuracy >= threshold
        cond_gte_accuracy = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report, 
                json_path="best_model.accuracy"
            ),
            right=self.accuracy_threshold
        )
        
        # Choose best model for registration
        best_model_name = JsonGet(
            step_name=evaluation_step.name,
            property_file=evaluation_report,
            json_path="best_model.name"
        )
        
        # Create registration steps for both models
        xgb_register = self.create_model_registration_step(xgb_step, evaluation_step, evaluation_report)
        lr_register = self.create_model_registration_step(lr_step, evaluation_step, evaluation_report)
        
        # Conditional step
        conditional_step = ConditionStep(
            name="ConditionalModelRegistration",
            conditions=[cond_gte_accuracy],
            if_steps=[xgb_register],  # In practice, you'd choose based on best_model_name
            else_steps=[]
        )
        
        return conditional_step
    
    def create_pipeline(self):
        """
        Create the complete SageMaker Pipeline
        """
        
        # Create all pipeline steps
        preprocessing_step = self.create_preprocessing_step()
        
        # Parallel training steps
        xgb_training_step = self.create_xgboost_training_step(preprocessing_step)
        lr_training_step = self.create_logistic_regression_training_step(preprocessing_step)
        
        # Evaluation step
        evaluation_step, evaluation_report = self.create_evaluation_step(
            preprocessing_step, xgb_training_step, lr_training_step
        )
        
        # Conditional registration
        conditional_registration = self.create_conditional_registration_step(
            evaluation_step, evaluation_report, xgb_training_step, lr_training_step
        )
        
        # Define the pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                self.input_data_uri,
                self.output_prefix,
                self.model_approval_status,
                self.accuracy_threshold,
                self.processing_instance_type,
                self.training_instance_type,
                self.instance_count,
                self.xgb_max_depth,
                self.xgb_eta, 
                self.xgb_num_round,
                self.lr_c,
                self.lr_max_iter
            ],
            steps=[
                preprocessing_step,
                xgb_training_step,
                lr_training_step, 
                evaluation_step,
                conditional_registration
            ],
            sagemaker_session=self.pipeline_session
        )
        
        return pipeline
    
    def deploy_pipeline(self):
        """
        Deploy the pipeline to SageMaker
        """
        
        pipeline = self.create_pipeline()
        
        # Create or update pipeline
        pipeline.upsert(role_arn=self.role)
        
        print(f"✅ Pipeline '{self.pipeline_name}' deployed successfully!")
        print(f"📊 Pipeline ARN: {pipeline.arn}")
        print(f"💰 Optimized for AWS Free Tier with cost controls")
        
        return pipeline
    
    def execute_pipeline(self, execution_display_name: str = None):
        """
        Execute the pipeline
        """
        
        if not execution_display_name:
            execution_display_name = f"olink-execution-{int(time.time())}"
        
        pipeline = self.create_pipeline()
        
        execution = pipeline.start(
            execution_display_name=execution_display_name
        )
        
        print(f"🚀 Pipeline execution started: {execution.arn}")
        print(f"📈 Monitor progress in SageMaker Console")
        
        return execution

# Example usage
if __name__ == "__main__":
    import time
    
    # Create and deploy pipeline
    olink_pipeline = OLINKPipeline("config/free_tier_config.json")
    
    # Deploy pipeline
    pipeline = olink_pipeline.deploy_pipeline()
    
    # Execute pipeline
    execution = olink_pipeline.execute_pipeline("olink-free-tier-demo")
    
    print("🎯 Pipeline deployed and executed successfully!")