import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import from your structure: src/pipeline/pipeline_def.py
    from src.pipeline.pipeline_def import OLINKPipeline
    import sagemaker
    import boto3
    print("Successfully imported pipeline definition")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have __init__.py files in src/ and src/pipeline/ folders")
    print("Current working directory:", os.getcwd())
    print("Project root:", project_root)
    sys.exit(1)

def main():
    print("OLINK COVID-19 Classification Pipeline Launcher")
    print("=" * 60)
    print(f"Launch time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Setup SageMaker session and role
        session = sagemaker.Session()
        role = os.getenv("SAGEMAKER_ROLE")
        
        if not role:
            print("⚠️  SAGEMAKER_ROLE not found in environment, auto-detecting...")
            role = session.get_caller_identity_arn()
            print(f"✅ Auto-detected role: {role}")
        else:
            print(f"✅ Using environment role: {role}")
        
        # Create pipeline
        print("\n📋 Creating pipeline...")
        olink_pipeline = OLINKPipeline("config/free_tier_config.json")
        print(f"✅ Pipeline configured with bucket: {olink_pipeline.bucket}")
        
        # Build pipeline definition
        print("🏗️  Building pipeline definition...")
        pipeline = olink_pipeline.create_pipeline()
        print("✅ Pipeline definition created successfully")
        
        # Deploy pipeline
        print("📤 Deploying pipeline to SageMaker...")
        pipeline.upsert(role_arn=role)
        print("✅ Pipeline deployed successfully")
        
        # Start execution
        print("🚀 Starting pipeline execution...")
        execution = pipeline.start()
        
        print("\n" + "=" * 60)
        print("🎉 PIPELINE LAUNCHED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📊 Execution ARN: {execution.arn}")
        print(f"📈 Pipeline Name: {olink_pipeline.pipeline_name}")
        print(f"🪣 S3 Bucket: {olink_pipeline.bucket}")
        print(f"🌍 Region: {olink_pipeline.region}")
        print("\n📱 Monitor Progress:")
        print(f"   AWS Console: https://console.aws.amazon.com/sagemaker/home?region={olink_pipeline.region}#/pipelines")
        print("\n📁 Expected Outputs (after completion):")
        print(f"   s3://{olink_pipeline.bucket}/pipeline-output/processed/ (preprocessed data)")
        print(f"   s3://{olink_pipeline.bucket}/pipeline-output/evaluation/ (model comparison)")
        print(f"   s3://{olink_pipeline.bucket}/pipeline-output/plots/ (visualizations)")
        print("\n⏱️  Estimated completion time: 15-30 minutes")
        print("=" * 60)
        
        return execution.arn
        
    except Exception as e:
        print(f"\n❌ Pipeline launch failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    execution_arn = main()
    if execution_arn:
        print(f"\n💡 To monitor progress, run:")
        print(f"   python3 scripts/monitor_pipeline.py {execution_arn}")
    else:
        sys.exit(1)