from unsloth import FastLanguageModel

def merge_lora_weights():
    """
    Merges the LoRA weights with the base model and saves the merged model
    in the Hugging Face format.
    """
    
    # The name of the base model from the Hugging Face Hub.
    base_model_name = "unsloth/gemma-3-1b-it"

    # The path to the LoRA checkpoint directory.
    # You can change this to the desired checkpoint.
    lora_checkpoint_dir = "outputs (og)/checkpoint-45735"
    
    # The directory where the merged model will be saved.
    output_dir = "merged_model"

    print(f"Loading base model: {base_model_name}")
    
    # Load the base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = base_model_name,
        max_seq_length = 8192,
        dtype = None,
        load_in_4bit = True,
    )

    print(f"Merging LoRA adapter from: {lora_checkpoint_dir}")

    # Unsloth's `from_pretrained` can directly load and merge a LoRA adapter.
    # We do this by specifying the model name as the LoRA checkpoint directory.
    # The library is smart enough to load the base model from the `adapter_config.json`
    # and then merge the adapter weights.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = lora_checkpoint_dir,
        max_seq_length = 8192,
        dtype = None,
        load_in_4bit = True,
    )
    
    print("Saving merged model...")
    model.save_pretrained_merged(
        output_dir, 
        tokenizer, 
        save_method="merged_16bit",
    )
    print(f"Merge completed! Merged model saved to {output_dir}")

if __name__ == "__main__":
    merge_lora_weights()