import json
import os

def load_data_from_s3():

    # TODO: handling it in S3

    # TODO: Temporary - get it locally
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(cur_dir, 'shy_ai_conversations.json')
    if not os.path.exists(file_path):
        raise FileNotFoundError("The file 'shy_ai_conversations.json' does not exist.")
    with open(file_path, "r", encoding="utf-8") as file:
        custom_data = json.load(file)

    return custom_data