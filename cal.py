import subprocess
import sys

def query_model(prompt):
    """Query the GGUF model and return formatted output"""
    
    # Build the full prompt with formatting
    full_prompt = f"### Instruction: {prompt}\n### Output:"
    
    cmd = [
        r".\llama-cli.exe",
        "-m", r"D:\PRATHMESH NIKAM\Downloads\VS\llm-fine-tuning\merged_model.Q8_0.gguf",
        "-ngl", "27",
        "-n", "512",
        "-p", full_prompt,
        "-no-cnv",  # Changed from --no-cnv to -no-cnv
        "--temp", "0.7",
        "-r", "<|endoftext|>"
    ]
    
    print(f"Querying model with prompt: {prompt}\n")
    print("=" * 60)
    
    try:
        # Run the command
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=r"D:\PRATHMESH NIKAM\Downloads\VS\llama-cli-cuda",
            timeout=60
        )
        
        output = result.stdout
        
        # Find the output section
        if "### Output:" in output:
            # Extract code after "### Output:"
            parts = output.split("### Output:")
            if len(parts) > 1:
                code = parts[-1].split("<|endoftext|>")[0]
                
                # Clean up the output
                code = code.strip()
                # Remove leading/trailing quotes if present
                code = code.strip('"')
                # Replace escaped newlines with actual newlines
                code = code.replace('\\n', '\n')
                code = code.replace('\\t', '    ')  # Replace tabs
                
                print("Generated Code:")
                print("-" * 60)
                print(code)
                print("-" * 60)
                return code
        
        # If parsing failed, show what we got
        if output:
            print("Model output:")
            print(output)
        
        if result.stderr:
            print("\nDebug info (stderr):")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("Error: Command timed out after 60 seconds")
    except Exception as e:
        print(f"Error running command: {e}")
    
    return None

if __name__ == "__main__":
    # Test prompt
    prompt = "Write a Python calculator that takes two numbers and an operation from the user"
    
    code = query_model(prompt)
    
    if code:
        print("\n✓ Code generation successful!")
        
        # Optionally save to file
        with open("generated_code.py", "w", encoding='utf-8') as f:
            f.write(code)
        print("✓ Saved to generated_code.py")
    else:
        print("\n✗ Code generation failed or needs manual extraction from output")
