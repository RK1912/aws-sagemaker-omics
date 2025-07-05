# Add this import at the top with your other imports
from sagemaker.processing import ScriptProcessor

class OLINKPipeline:
    """
    Enhanced SageMaker Pipeline for OLINK COVID-19 Classification
    Optimized for AWS Free Tier with cost controls
    """
    
    def __init__(self, config_path: str = "config/free_tier_config.json"):
        self.config = self._load_config(config_path)
        self.region = self.config['aws']['region']
        self.role = os.getenv(self.config['aws']['sagemaker_role'])
        self.bucket = self.config['aws']['bucket']
        self.pipeline_name = self.config['pipeline']['name']
        
        # Initialize SageMaker session
        self.pipeline_session = PipelineSession()
        self.sagemaker_session = sagemaker.Session()
        
        # Define pipeline parameters
        self._define_parameters()
    
    
    def create_preprocessing_step(self):
        """
        Create data preprocessing step
        UPDATED: Use ScriptProcessor with public image for matplotlib support
        """
        
        # Replace SKLearnProcessor with ScriptProcessor using public image
        script_processor = ScriptProcessor(
            image_uri=os.getenv('SAGEMAKER_IMAGE_URI', 'public.ecr.aws/sagemaker/sagemaker-distribution:3.1-cpu'),
            command=["python3"],
            instance_type=self.processing_instance_type,
            instance_count=self.instance_count,
            base_job_name="olink-preprocessing",
            role=self.role,
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=3600  # 1 hour limit for cost control
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
    
    
    def create_evaluation_step(self, preprocessing_step, xgb_step, lr_step):
        """
        Create model evaluation step that compares both models
        UPDATED: Use ScriptProcessor with public image for matplotlib support
        """
        
        # Replace SKLearnProcessor with ScriptProcessor using public image
        eval_processor = ScriptProcessor(
            image_uri=os.getenv('SAGEMAKER_IMAGE_URI', 'public.ecr.aws/sagemaker/sagemaker-distribution:3.1-cpu'),
            command=["python3"],
            instance_type=self.processing_instance_type,
            instance_count=self.instance_count,
            base_job_name="olink-model-evaluation",
            role=self.role,
            sagemaker_session=self.pipeline_session,
            max_runtime_in_seconds=1800  # 30 minutes
        )
        
        # Everything else stays exactly the same
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation", 
            path="evaluation.json"
        )
        
        evaluation_step = ProcessingStep(
            name="EvaluateModels",
            processor=eval_processor,  #use eval_processor instead of sklearn processor
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
    