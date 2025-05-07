from s3_handler.load_data import load_data_from_s3
from ai_training.training import start_training

if __name__ == "__main__":
    start_training()
    #load_data_from_s3()