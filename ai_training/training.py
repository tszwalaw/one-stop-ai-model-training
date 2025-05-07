from unsloth import FastLlamaModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from s3_handler.load_data import load_data_from_s3
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

from functools import partial
import torch

def start_training():

    custom_data = load_data_from_s3()

    # AI Model config
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # Loading the pre-train model
    model, tokenizer = FastLlamaModel.from_pretrained(
        model_name = "unsloth/Llama-3.2-3B-Instruct",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # Setting up LoRA Fine-tuning configuration
    '''
    Available components:
    q_proj	    Projects input to query vectors (for attention scoring)
    k_proj	    Projects input to key vectors (paired with queries)
    v_proj	    Projects input to value vectors (used for weighted sum)
    o_proj	    Projects the output of attention back to the original dimension
    gate_proj	Often used in gated activations (like SwiGLU); projects input to a gated space
    up_proj	    Projects to a higher dimensional space in FFN
    down_proj	Projects back to original dimension after nonlinear transformation
    
    How to choose:
    Fine-tune only q_proj, v_proj: 
    - Focuses on modifying attention behavior with minimal overhead.
    Include MLPs (gate_proj, etc.): 
    - Gives more flexibility to transform internal representations, but adds more parameters.
    Fine-tune all (q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj): 
    - Maximum expressiveness, higher memory/computation cost.
    '''
    model = FastLlamaModel.get_peft_model(
        model,
        r = 16, # Set the LoRA rank (determines the complexity of adaptations)

        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16, # Sets the scaling factor for LoRA updates
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ quantization config
    )

    # Put Data with the AI Model
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3.1",
    )
    dataset = standardize_sharegpt(custom_data)
    formatted_map_function = partial(formatting_prompts_func, tokenizer)
    dataset = dataset.map(formatted_map_function, batched = True)

    # Start training
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # num_train_epochs = 1, # Set this for 1 full training run.
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        ),
    )
    trainer_stats = trainer.train()
    model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")



    # TODO: Upload AI Model to S3 via S3 Helper


    return "completed"

def formatting_prompts_func(tokenizer, data):
    convos = data["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }