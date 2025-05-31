import requests
import json
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def test_single_search():
    test_query = "Người sử dụng lao động được sa thải người lao động nữ đang mang thai không?"
    
    
    payload = {
        "query": test_query,
        "top_k": 3
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(data)
              
        else:
            print(f"Search failed with status: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Search error: {e}")

if __name__ == "__main__":
    test_single_search()