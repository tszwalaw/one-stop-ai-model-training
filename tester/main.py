from s3_handler.load_data import load_data_from_s3
from ai_training.training_local_ai import start_training

if __name__ == "__main__":
    hf_token = "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"  # Replace with your token or set HF_TOKEN environment variable
    start_training(hf_token=hf_token)