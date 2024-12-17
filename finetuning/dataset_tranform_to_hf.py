import json
import pandas as pd

def process_json_dataset(input_path: str, output_path: str) -> None:
    """
    Process JSON dataset and convert it to unsloth format.
    """
    # Read the JSON file
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Extract QA pairs
    formatted_data = []
    for qa in data['qa_pairs']:
        formatted_data.append({
            'instruction': qa['question'],
            'output': qa['answer'],
            'input': ''  # Empty input as per unsloth format
        })
    
    # Convert to DataFrame and save as JSONL
    df = pd.DataFrame(formatted_data)
    df.to_json(output_path, orient='records', lines=True)
    
    print(f"Processed {len(df)} samples and saved to {output_path}")
    print("\nSample of processed data:")
    print(df.head().to_string())

# Example usage
if __name__ == "__main__":
    process_json_dataset('curated_search/qa_pairs_curated.json', 'processed_dataset.json')