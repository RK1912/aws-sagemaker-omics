# train.py
import time

def mock_training():
    print("Starting mock training job...")
    for i in range(5):
        print(f"Training step {i + 1}/5")
        time.sleep(1)
    print("Training completed!")

if __name__ == "__main__":
    mock_training()
    print("Build and train.py executed successfully!")

