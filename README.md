# 🤖 One-Stop AI Model Training Service

This project provides a streamlined, end-to-end pipeline for training and deploying AI language models using custom datasets. Users can upload data, trigger training, and deploy the resulting model as an API—all integrated with AWS infrastructure.

---

## 📌 Overview

This service enables users to:

1. **Upload custom training data to S3**
2. **Fine tune an AI model** based on the uploaded dataset
3. **Package the model as a GGUF format and build a FastAPI-based inference server**
4. **Deploy the server to EC2 using infrastructure as code**

## 🛠 Technologies Used

- Python for model training and API logic
- HuggingFace Transformers / GGUF for fine-tuning and inference
- AWS S3 & EC2 for data and deployment
- Docker for packaging
- FastAPI for serving the model
- Terraform for infrastructure deployment

## 🔁 Workflow
1. **Data Ingestion**
    - User uploads a JSON file (formatted for fine-tuning) to a specific S3 bucket.
    - File format example:
      ```json
      {
        "conversations": [
          [
            {"role": "system", "content": "You are an assistant"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It's 4."}
          ]
        ]
      }
      ```
      
2. **Model Training**
    - The training service pulls the dataset from S3.
    - Initializes and fine-tunes a base language model using the uploaded data.
    - Converts the trained model to `GGUF` format.
    - Uploads the GGUF model back to S3.

3. **Model Deployment**
    - Builds a Docker image containing:
        - The GGUF model
        - A FastAPI app for inference
    - Pushes the image to a container registry 
    - Uses Terraform or another IaC tool (feel free to contribute) to deploy the image to an EC2 instance for hosting.

## Hardware Requirements
- **Required GPU for accelerated model training**

