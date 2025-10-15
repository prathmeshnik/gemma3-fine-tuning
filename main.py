import json
import torch
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os
from datetime import datetime

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
    with open(file_path, 'r', encoding="utf-8") as f:
        file_data = json.load(f)
    
    print(f"Dataset sample: {file_data[1] if len(file_data) > 1 else file_data[0]}")
    
    # The JSON file already contains a 'prompt' field with the formatted text.
    # We can use it directly. The SFTTrainer will handle the EOS token.
    formatted_data = [item['prompt'] for item in file_data]
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
    
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 8
        warmup_steps=10,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        logging_strategy="steps",
        logging_dir="./logs",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        save_strategy="epoch",
        save_total_limit=3,
        dataloader_pin_memory=False,
        report_to="tensorboard",
    )
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,
        args=training_args,
    )
    
    trainer_stats = trainer.train()
    print("Training completed!")
    return trainer_stats, training_args

def visualize_generation(model, tokenizer):
    """Generate sample responses from the model"""
    print("\n\n--- Visualizing Model Generation ---")
    
    model.eval()
    
    prompts = [
        "Write a Python function to find the factorial of a number.",
        "Explain the difference between a list and a tuple in Python.",
        "Create a simple HTML page with a button."
    ]
    
    prompt_template = '''Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
'''
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for instruction in prompts:
        prompt = prompt_template.format(instruction, "")
        inputs = tokenizer([prompt], return_tensors="pt").to(device)
        
        outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True, eos_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # Extract only the response part
        response = decoded_output[len(prompt):]
        clean_response = response.split("<|endoftext|>")[0].strip()

        print("-" * 30)
        print(f"Instruction: {instruction}")
        print(f"Response:\n{clean_response}")
        print("-" * 30)

def generate_training_report(trainer_stats, training_args, model_config, dataset_path):
    """Generate and save a report of the training process"""
    print("\n--- Generating Training Report ---")
    
    # Safely convert training_args to a dictionary
    try:
        training_args_dict = training_args.to_dict()
    except Exception:
        training_args_dict = {}

    report = f"""
# Training Report

## 1. Overview
This report summarizes the fine-tuning process of the language model.

- **Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Training Duration:** {getattr(trainer_stats, 'training_duration', 'N/A'):.2f} seconds
- **Training Loss:** {getattr(trainer_stats, 'training_loss', 'N/A'):.4f}

## 2. Model Configuration
- **Base Model:** {model_config.get('model_name', 'N/A')}
- **Max Sequence Length:** {model_config.get('max_seq_length', 'N/A')}
- **Quantization:** 4-bit

### LoRA Configuration
```json
{json.dumps(model_config.get('lora_config', {}), indent=4)}
```

## 3. Dataset
- **Dataset Path:** `{dataset_path}`

## 4. Training Arguments
```json
{json.dumps(training_args_dict, indent=4, default=str)}
```

## 5. Environment
- **CUDA available:** {torch.cuda.is_available()}
- **GPU:** {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}
- **PyTorch Version:** {torch.__version__}

---
*This report is saved as `training_report.md`.*
*To visualize training metrics, run: `tensorboard --logdir=./logs`*
"""
    
    report_md = report.replace("```json", "```\n").replace("```", "\n```")
    with open("training_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
        
    print(report)

def main():
    # Check GPU
    check_gpu()
    
    # Load dataset
    dataset_path = "dataset/code_instructions_1k_alpaca_subset.json"
    dataset = load_dataset(dataset_path)
    
    # Load model
    model, tokenizer, max_seq_length = load_model()
    
    # Model config for report
    model_config = {
        "model_name": "unsloth/gemma-3-1b-it",
        "max_seq_length": max_seq_length,
        "lora_config": {
            "r": 64,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_alpha": 128,
            "lora_dropout": 0,
            "bias": "none",
        }
    }
    
    # Setup LoRA
    model = setup_lora(model)
    
    # Train the model
    trainer_stats, training_args = train_model(model, tokenizer, dataset, max_seq_length)
    
    print("Training complete! Model saved in 'outputs' directory.")
    
    # Generate report and visualize
    generate_training_report(trainer_stats, training_args, model_config, dataset_path)
    visualize_generation(model, tokenizer)

if __name__ == "__main__":
    main()