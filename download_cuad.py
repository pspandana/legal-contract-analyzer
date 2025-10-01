"""
Download CUAD dataset from HuggingFace and convert to expected format
"""
from datasets import load_dataset
import json
from pathlib import Path

print("Downloading CUAD dataset from HuggingFace...")

# Correct dataset name - it's stored as a community dataset
try:
    # Try the correct path
    dataset = load_dataset("coastalcph/CUAD")
    
    print(f"Downloaded! Processing {len(dataset['train'])} contracts...")
    
    # Create data directory
    Path("data").mkdir(exist_ok=True)
    
    # Convert to format your code expects
    cuad_data = {
        "version": "aok_v1.0",
        "data": []
    }
    
    # Group by contract title since CUAD has one row per question
    from collections import defaultdict
    contracts = defaultdict(lambda: {"title": None, "paragraphs": []})
    
    for item in dataset['train']:
        title = item['title']
        
        if contracts[title]["title"] is None:
            contracts[title]["title"] = title
        
        # Create paragraph with context and QA
        paragraph = {
            "context": item['context'],
            "qas": [{
                "id": item['id'],
                "question": item['question'],
                "answers": {
                    "text": item['answers']['text'],
                    "answer_start": item['answers']['answer_start']
                },
                "is_impossible": item.get('is_impossible', False)
            }]
        }
        
        contracts[title]["paragraphs"].append(paragraph)
    
    # Convert to list
    cuad_data["data"] = list(contracts.values())
    
    # Save
    output_file = Path("data/CUADv1.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cuad_data, f, indent=2)
    
    print(f"✓ Saved {len(cuad_data['data'])} contracts to {output_file}")
    print(f"✓ File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
    print("\nYou can now run: streamlit run app.py")
    
except Exception as e:
    print(f"\nDownload failed: {e}")
    print("\nAlternative: Download manually from GitHub")
    print("1. Go to: https://github.com/TheAtticusProject/cuad")
    print("2. Download data/CUAD_v1.json")
    print("3. Save to: data/CUADv1.json in your project")