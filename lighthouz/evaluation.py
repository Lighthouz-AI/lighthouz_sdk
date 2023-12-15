from typing import Optional, Callable, List

import requests
from marshmallow import ValidationError

from lighthouz.schema import benchmark_schema



class Evaluation:
    def __init__(self, LH):
        self.LH = LH

    def evaluate_rag_model(
        self,
        response_function: Callable[[str], str],
        benchmark_id: Optional[str] = None,
        benchmarks: Optional[List[dict]] = None,
    ):
        test_create_url = f"{self.LH.base_url}/apps/{self.LH.lh_app_id}/tests/create"
        test_create_data = {"status": "running"}
        test_create_headers = {
            "api-key": self.LH.lh_api_key,
        }
        test_create_response = requests.post(
            test_create_url, headers=test_create_headers, json=test_create_data
        )
        if test_create_response.status_code == 200:
            test_id = test_create_response.json()["test_id"]
            if benchmark_id:
                benchmark_url = f"{self.LH.base_url}/apps/benchmarks/{benchmark_id}"
                benchmark_headers = {
                    "api-key": self.LH.lh_api_key,
                }
                benchmark_response = requests.get(
                    benchmark_url, headers=benchmark_headers
                )
                if benchmark_response.status_code == 200:
                    benchmarks = benchmark_response.json()["benchmark"]["benchmark"]
                else:
                    return {
                        "success": False,
                        "message": benchmark_response.json()["message"],
                    }
            else:
                if not benchmarks:
                    return {
                        "success": False,
                        "message": "Either benchmark_id or benchmark must be provided",
                    }
            try:
                for benchmark in benchmarks:
                    print(benchmark)
                    benchmark_schema.load(benchmark)
            except ValidationError as err:
                return {"success": False, "message": err.messages}

            for benchmark in benchmarks:
                benchmark["generated_response"] = response_function(benchmark["query"])
            print(benchmarks)
            evaluation_url = f"{self.LH.base_url}/api/{test_id}/docqa_evaluate_group"
            evaluation_data = benchmarks
            evaluation_headers = {
                "api-key": self.LH.lh_api_key,
            }
            evaluation_response = requests.post(
                evaluation_url, headers=evaluation_headers, json=evaluation_data
            )
            if evaluation_response.status_code == 200:
                evaluation = evaluation_response.json()
                return evaluation
            else:
                print("error")
                return {
                    "success": False,
                    "message": evaluation_response.json()["message"],
                }
