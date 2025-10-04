import torch
from unsloth import FastLanguageModel

def test_fine_tuned_model():
    """Test the fine-tuned model"""
    print("Loading fine-tuned model for testing...")
    
    # Load the saved model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="outputs/checkpoint-9306",
        max_seq_length=8192,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    # Test prompt
    messages = [
        {"role": "user", "content": "Calculate how much time I spend on my phone per week!, how can i calcuate it using python code"},
    ]
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Generating response...")
    
    # Generate response - extract input_ids and attention_mask from dict
    outputs = model.generate(
        input_ids=inputs["input_ids"],  # Extract from dictionary
        attention_mask=inputs["attention_mask"],  # Extract from dictionary
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode and print
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    print("Model Response:")
    print(response.replace('\\n', '\n'))

if __name__ == "__main__":
    test_fine_tuned_model()
