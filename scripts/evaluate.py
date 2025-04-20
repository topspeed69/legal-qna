import argparse
import yaml
import logging.config
import json
from pathlib import Path
import sys
import os
import requests
from typing import Dict, List
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)

def setup_logging():
    with open("config/logging.yaml") as f:
        logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)

def load_test_cases(file_path: str) -> List[Dict]:
    with open(file_path) as f:
        return json.load(f)

def evaluate_response(response: Dict, expected: Dict) -> Dict:
    # Basic evaluation - can be extended with more sophisticated metrics
    metrics = {
        "has_answer": bool(response.get("answer")),
        "has_sources": bool(response.get("sources")),
        "response_time": response.get("response_time", 0),
    }
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate QA system performance")
    parser.add_argument("--test-file", required=True, help="JSON file containing test cases")
    parser.add_argument("--output-file", required=True, help="File to write evaluation results")
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    api_url = f"http://{config['api']['host']}:{config['api']['port']}/api/query"
    
    test_cases = load_test_cases(args.test_file)
    results = []
    
    for test_case in test_cases:
        try:
            start_time = time.time()
            response = requests.post(
                api_url,
                json={"question": test_case["question"]}
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                result["response_time"] = response_time
                metrics = evaluate_response(result, test_case.get("expected", {}))
                
                results.append({
                    "test_case": test_case,
                    "response": result,
                    "metrics": metrics
                })
                
                logger.info(f"Processed test case: {test_case['question']}")
            else:
                logger.error(f"Error processing test case: {response.text}")
                
        except Exception as e:
            logger.error(f"Error evaluating test case: {e}")
    
    # Calculate aggregate metrics
    aggregate_metrics = {
        "total_cases": len(test_cases),
        "successful_responses": sum(1 for r in results if r["metrics"]["has_answer"]),
        "average_response_time": sum(r["metrics"]["response_time"] for r in results) / len(results) if results else 0
    }
    
    # Save results
    output = {
        "aggregate_metrics": aggregate_metrics,
        "detailed_results": results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Evaluation complete. Results written to {args.output_file}")

if __name__ == "__main__":
    main()