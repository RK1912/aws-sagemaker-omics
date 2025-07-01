# monitor_pipeline.py
import boto3
import time
from datetime import datetime

def monitor_pipeline_execution(execution_arn):
    """Monitor pipeline execution status"""
    
    sagemaker = boto3.client('sagemaker')
    
    print(f"🔍 Monitoring pipeline execution...")
    print(f"Execution ARN: {execution_arn}")
    print("=" * 60)
    
    while True:
        try:
            response = sagemaker.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            
            status = response['PipelineExecutionStatus']
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] Status: {status}")
            
            if status in ['Succeeded', 'Failed', 'Stopped']:
                print(f"\n🏁 Pipeline completed with status: {status}")
                
                if status == 'Succeeded':
                    print("✅ Pipeline executed successfully!")
                    print("📁 Check your S3 bucket for outputs:")
                    print(f"   s3://omics-ml/pipeline-output/")
                else:
                    print("❌ Pipeline failed or was stopped")
                    print("📝 Check CloudWatch logs for details")
                
                break
                
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⚠️ Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            break

if __name__ == "__main__":
    # Replace with your actual execution ARN
    execution_arn = input("Enter execution ARN: ").strip()
    monitor_pipeline_execution(execution_arn)