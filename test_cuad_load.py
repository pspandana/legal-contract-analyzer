import json
from pathlib import Path

def test_cuad_loading():
    cuad_file = Path("data/CUADv1.json")
    
    if not cuad_file.exists():
        print("CUAD data not found! Need to download it.")
        return False
    
    try:
        with open(cuad_file, 'r', encoding='utf-8') as f:
            cuad_data = json.load(f)
        
        print(f"✓ CUAD loaded successfully")
        print(f"✓ Version: {cuad_data.get('version', 'Unknown')}")
        print(f"✓ Number of contracts: {len(cuad_data['data'])}")
        
        # Test first contract structure
        first_contract = cuad_data['data'][0]
        print(f"✓ First contract: {first_contract['title']}")
        print(f"✓ Number of Q&A pairs: {len(first_contract['paragraphs'][0]['qas'])}")
        
        return True
        
    except Exception as e:
        print(f"Error loading CUAD: {e}")
        return False

if __name__ == "__main__":
    test_cuad_loading()