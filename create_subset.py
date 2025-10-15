import json
import random

def create_subset(input_file, output_file, subset_size=1000):
    """
    Creates a subset of the given JSON dataset file.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to the output JSON file
        subset_size (int): Number of items to include in the subset
    """
    # Load the full dataset
    with open(input_file, 'r', encoding='utf-8') as f:
        full_dataset = json.load(f)
    
    # Create a random subset
    if len(full_dataset) <= subset_size:
        subset = full_dataset
    else:
        subset = random.sample(full_dataset, subset_size)
    
    # Write the subset to the output file with proper formatting
    with open(output_file, 'w', encoding='utf-8') as f:
        # Write opening bracket
        f.write('[\n')
        
        # Write each item in the subset
        for i, item in enumerate(subset):
            # Write the JSON item
            json.dump(item, f, ensure_ascii=False, indent=2)
            
            # Add comma after each item except the last one
            if i < len(subset) - 1:
                f.write(',')
            
            # Add newline after each item
            f.write('\n')
        
        # Write closing bracket
        f.write(']\n')
    
    print(f"Subset of {len(subset)} items created successfully in {output_file}")

if __name__ == "__main__":
    input_path = "dataset/code_instructions_120k_alpaca.json"
    output_path = "dataset/code_instructions_1k_alpaca_subset.json"
    
    # Create a subset of 1000 items
    create_subset(input_path, output_path, 1000)