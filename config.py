"""
Configuration settings for the fine-tuning process
"""

# Model settings
MODEL_NAME = "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048

# LoRA settings
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0

# Training settings
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10

# Paths
DATASET_PATH = "json_extraction_dataset_500.json"
OUTPUT_DIR = "outputs"
GGUF_OUTPUT_DIR = "gguf_model"
