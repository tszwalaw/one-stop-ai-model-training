import os
import time
from datetime import datetime
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from huggingface_hub import snapshot_download
from s3_handler.load_data import load_data_from_s3

# Define the function to preprocess and tokenize the dataset
def preprocess_data(data, tokenizer):
    """
    Preprocess the input dataset into a format suitable for instruction tuning.
    Applies LLaMA's chat template to prompt-response pairs.
    """
    processed_data = []
    for conversation in data:
        chat = [
            {"role": "user", "content": conversation['prompt']},
            {"role": "assistant", "content": conversation['response']}
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        processed_data.append(text)
    return processed_data

def tokenize_data(processed_data, tokenizer):
    """
    Tokenizes the preprocessed data and pads/truncates to a fixed length.
    """
    return tokenizer(processed_data, padding=True, truncation=True, max_length=512)

def is_valid_model_dir(model_dir):
    """
    Check if the model directory exists and contains required files.
    """
    required_files = ["config.json", "tokenizer.json"]
    if not os.path.isdir(model_dir):
        return False
    return any(os.path.isfile(os.path.join(model_dir, f)) for f in required_files)

def create_output_dir(base_output_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_training_dir = os.path.join(base_output_dir,timestamp)

    adapter_weights_dir = os.path.join(new_training_dir, "adapter_weights")
    merged_model_dir = os.path.join(new_training_dir, "merged_model")

    os.makedirs(adapter_weights_dir, exist_ok=True)
    os.makedirs(merged_model_dir, exist_ok=True)

    return adapter_weights_dir, merged_model_dir

def start_training(
    local_model_dir="./llama-3.2-3b-instruct",
    output_dir="./fine-tuned-llama32",
    hf_token=None
):

    # Check if local model directory is valid
    model_dir = local_model_dir
    use_local_model = is_valid_model_dir(local_model_dir)

    # Load Hugging Face token from environment or parameter
    if not use_local_model:
        if not hf_token:
            raise ValueError("Please provide a Hugging Face token")
        print(f"Local model directory {local_model_dir} is empty or invalid. Downloading meta-llama/Llama-3.2-3B-Instruct")
        model_dir = "meta-llama/Llama-3.2-3B-Instruct"
        snapshot_download(
            repo_id=model_dir,
            local_dir=local_model_dir,
            token=hf_token
        )
        use_local_model = True  # Now use the downloaded local files
        model_dir = local_model_dir  # Switch to local directory for loading

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token 

    # Load dataset from S3 if no local dataset path is provided
    # TODO: handle the data in streaming
    data = load_data_from_s3()

    # Preprocess and tokenize the data
    processed_data = preprocess_data(data, tokenizer)
    tokenized_data = tokenize_data(processed_data, tokenizer)

    # Convert to a Hugging Face Dataset
    dataset = Dataset.from_dict({
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["input_ids"]  # For causal LM, labels are the same as input_ids
    })

    # Load the pre-trained LLaMA model
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token if not use_local_model else None
)

    # Configure LoRA for parameter-efficient fine-tuning
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        gradient_accumulation_steps=4
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset  # Replace with a separate validation dataset if available
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model (adapter weights)
    adapter_weights_dir, merged_model_dir = create_output_dir(output_dir)

    # Only the LoRA adapters (delta changes)
    trainer.save_model(adapter_weights_dir)
    tokenizer.save_pretrained(adapter_weights_dir)
    print(f"Adapter weights saved to {adapter_weights_dir}")

    # Merge LoRA adapters with base model for deployment
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_model_dir)
    tokenizer.save_pretrained(merged_model_dir)
    print(f"Merged model saved to {merged_model_dir}")
