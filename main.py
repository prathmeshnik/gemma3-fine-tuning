import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

def check_gpu():
    """Check GPU availability"""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: None")

def load_dataset(file_path):
    """Load and format the dataset"""
    print(f"Loading dataset from {file_path}...")
    with open(file_path, 'r') as f:
        file_data = json.load(f)
    
    print(f"Dataset sample: {file_data[1] if len(file_data) > 1 else file_data[0]}")
    
    def format_prompt(example):
        # Using instruction, input, and output fields from your JSON structure
        return f"### Instruction: {example['instruction']}\\n### Input: {example['input']}\\n### Output: {json.dumps(example['output'])}<|endoftext|>"

    formatted_data = [format_prompt(item) for item in file_data]
    dataset = Dataset.from_dict({"text": formatted_data})
    
    print(f"Dataset size: {len(dataset)}")
    return dataset

def load_model():
    """Load the base model and tokenizer"""
    model_name = "unsloth/gemma-3-1b-it"
    max_seq_length = 2048
    dtype = None  # Auto detection
    
    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,
    )
    
    return model, tokenizer, max_seq_length

def setup_lora(model):
    """Add LoRA adapters to the model"""
    print("Setting up LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=64,  # LoRA rank - higher = more capacity, more memory
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=128,  # LoRA scaling factor (usually 2x rank)
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",     # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # Unsloth's optimized version
        random_state=3407,
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None, # LoftQ
    )
    return model

def train_model(model, tokenizer, dataset, max_seq_length):
    """Train the model"""
    print("Starting training...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,  # Effective batch size = 8
            warmup_steps=10,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=25,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            save_strategy="epoch",
            save_total_limit=2,
            dataloader_pin_memory=False,
            report_to="none",  # Disable Weights & Biases logging
        ),
    )
    
    trainer_stats = trainer.train()
    print("Training completed!")
    return trainer_stats

def main():
    # Check GPU
    check_gpu()
    
    # Load dataset (you'll need to provide this file)
    dataset = load_dataset("python-codes.json")
    
    # Load model
    model, tokenizer, max_seq_length = load_model()
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Train the model
    trainer_stats = train_model(model, tokenizer, dataset, max_seq_length)
    
    print("Training complete! Model saved in 'outputs' directory.")

if __name__ == "__main__":
    main()
