import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.trag_v2.csep_retriever_v2 import _parse_entities

def test_parse_entities():
    test_cases = [
        "JIRA-123, PR #102, feature-x",
        "NONE",
        "",
        '{"entities": ["JIRA-123", "PR #102"]}',
        "  JIRA-123, PR #102  "
    ]
    
    print("Testing _parse_entities:")
    for i, tc in enumerate(test_cases):
        res = _parse_entities(tc)
        print(f"Case {i+1}:")
        print(f"  Input:  {tc!r}")
        print(f"  Output: {res!r}")
        print("-" * 40)

if __name__ == "__main__":
    test_parse_entities()
