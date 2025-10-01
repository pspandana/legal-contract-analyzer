import sys
sys.path.append('src')
from cuad_analyzer import CUADAnalyzer

def debug_patterns():
    analyzer = CUADAnalyzer()
    
    print("=== CUAD KNOWLEDGE BASE DEBUG ===")
    print(f"Total categories: {len(analyzer.category_knowledge)}")
    
    # Check specific categories that should match our test
    test_categories = ['Document Name', 'Termination For Convenience', 'Uncapped Liability', 'Governing Law']
    
    for category in test_categories:
        if category in analyzer.category_knowledge:
            knowledge = analyzer.category_knowledge[category]
            print(f"\n--- {category} ---")
            print(f"Examples found: {len(knowledge['examples'])}")
            print(f"Patterns extracted: {len(knowledge['patterns'])}")
            
            # Show first few patterns
            for i, pattern in enumerate(knowledge['patterns'][:3]):
                print(f"  Pattern {i+1}: {pattern}")
        else:
            print(f"\n--- {category} --- NOT FOUND")
    
    # Test our sample contract text
    test_contract = """
    DISTRIBUTION AGREEMENT
    
    This agreement terminates at the convenience of either party
    with 30 days notice. Liability under this agreement is unlimited
    and uncapped. This agreement is governed by Delaware law.
    """
    
    print(f"\n=== TESTING CONTRACT ===")
    print("Contract text:", test_contract.lower())
    
    # Check if any patterns would match
    for category in test_categories:
        if category in analyzer.category_knowledge:
            patterns = analyzer.category_knowledge[category]['patterns']
            matches = []
            for pattern in patterns[:5]:
                if pattern in test_contract.lower():
                    matches.append(pattern)
            if matches:
                print(f"{category}: Found matches: {matches}")
            else:
                print(f"{category}: No pattern matches")

if __name__ == "__main__":
    debug_patterns()