import os
from unsloth import FastLanguageModel

def export_to_gguf():
    """Export the trained model to GGUF format"""
    print("Loading model for GGUF export...")
    
    # Load the trained model from checkpoint
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/checkpoint-625",
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Step 1: Merge LoRA with base model and save as 16bit
    print("Merging LoRA adapters with base model...")
    model.save_pretrained_merged(
        "merged_model", 
        tokenizer, 
        save_method="merged_16bit",
    )
    print("Merge completed!")
    
    # Step 2: Convert the merged model to GGUF q8_0
    print("Converting merged model to GGUF format (this takes 10-15 minutes)...")
    model.save_pretrained_gguf(
        "merged_model",  # Must point to the merged model folder from step 1
        tokenizer, 
        quantization_method="q8_0"
    )
    print("GGUF export completed!")
    
    # List the exported files
    if os.path.exists("merged_model"):
        gguf_files = [f for f in os.listdir("merged_model") if f.endswith(".gguf")]
        if gguf_files:
            print(f"\nExported GGUF files:")
            for file in gguf_files:
                file_path = os.path.join("merged_model", file)
                file_size = os.path.getsize(file_path) / (1024**3)
                print(f"  - {file} ({file_size:.2f} GB)")
        else:
            print("No GGUF files found in merged_model directory")

if __name__ == "__main__":
    export_to_gguf()
