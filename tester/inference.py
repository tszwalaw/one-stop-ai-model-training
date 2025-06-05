import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def inference_testing():
    # Load the fine-tuned model and tokenizer
    model_dir = "./llama_finetuned"  # Replace with your path
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)

    # Ensure the tokenizer has a pad token (needed for inference)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos_token if missing

    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Sample prompt for testing
    prompt = (
        "You are a shy, AI-based robot assistant. You are very knowledgeable about programming but get nervous talking to people.\n"
        "User: What is Java?\n"
        "Assistant:"
    )

    # Tokenize input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move inputs to the same device as the model

    # Generate output with the model
    outputs = model.generate(
        inputs["input_ids"],
        max_length=100,  # Adjust length as needed
        num_return_sequences=1,  # Number of samples to generate
        temperature=0.7,  # Adjust creativity level
        top_p=0.9,  # Sampling strategy for diverse outputs
        top_k=50,  # Top-k sampling
        do_sample=True,  # Enable sampling for diversity
    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")
