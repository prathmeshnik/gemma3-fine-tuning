import torch
from unsloth import FastLanguageModel

def main():
    """
    This function runs a continuous conversation with the fine-tuned model.
    """
    
    # 1. Load the fine-tuned model and tokenizer
    # This will load the base model and apply the LoRA adapters from the checkpoint.
    model_path = "merged_model"  # Path to the merged model directory
    print(f"Loading model from: {model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=8192, # context-window
        dtype=None,  # Autodetect
        load_in_4bit=True,
    )
    
    # Set the model to evaluation mode
    model.eval()

    # 2. Check for GPU and move model to GPU
    if torch.cuda.is_available():
        print("Using GPU:", torch.cuda.get_device_name(0))
        # Unsloth with load_in_4bit handles device placement, but we ensure inputs are on the GPU.
        device = "cuda"
    else:
        print("WARNING: GPU not available, using CPU.")
        device = "cpu"

    # 3. Prepare the prompt template
    # We will build a continuous conversation log.
    prompt_template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

{}"""

    conversation_history = ""

    print("\n\nModel loaded. Start conversation (type 'exit' or 'quit' to stop).")
    while True:
        instruction = input("You: ")
        if instruction.lower() in ['exit', 'quit']:
            break

        # Format the prompt with the current instruction and conversation history
        current_prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        full_prompt = prompt_template.format(conversation_history + current_prompt)


        # 4. Tokenize the input prompt
        inputs = tokenizer([full_prompt], return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # 5. Generate the output from the model
        print("\nGenerating response...")
        outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True, eos_token_id=tokenizer.eos_token_id)

        # 6. Decode only the newly generated tokens
        new_tokens = outputs[0, input_ids.shape[1]:]
        clean_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        clean_response = clean_response.split('<|endoftext|>')[0].strip()


        # Print the results
        print("="*50)
        print("### Model Response ###")
        print(clean_response)
        print("="*50)

        # Update conversation history
        conversation_history += f"### Instruction:\n{instruction}\n\n### Response:\n{clean_response}\n\n"

if __name__ == "__main__":
    main()