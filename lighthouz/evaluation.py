from typing import Callable

import requests
from marshmallow import ValidationError

from lighthouz.schema import benchmark_schema


class Evaluation:
    def __init__(self, LH):
        self.LH = LH

    def evaluate_rag_model(
        self, response_function: Callable[[str], str], benchmark_id: str
    ):
        test_create_url = f"{self.LH.base_url}/apps/{self.LH.lh_app_id}/tests/create"
        test_create_data = {"status": "completed", "benchmark_id": benchmark_id}
        test_create_headers = {
            "api-key": self.LH.lh_api_key,
        }
        test_create_response = requests.post(
            test_create_url, headers=test_create_headers, json=test_create_data
        )
        if test_create_response.status_code == 200:
            test_id = test_create_response.json()["test_id"]
            benchmark_url = f"{self.LH.base_url}/apps/benchmarks/{benchmark_id}"
            benchmark_headers = {
                "api-key": self.LH.lh_api_key,
            }
            benchmark_response = requests.get(benchmark_url, headers=benchmark_headers)
            if benchmark_response.status_code == 200:
                benchmarks = benchmark_response.json()["benchmark"]["benchmark"]
            else:
                return {
                    "success": False,
                    "message": benchmark_response.json()["message"],
                }
            try:
                for benchmark in benchmarks:
                    # print(benchmark)
                    benchmark_schema.load(benchmark)
            except ValidationError as err:
                return {"success": False, "message": err.messages}

            print(f"Evaluating on {len(benchmarks)} benchmark(s).")
            results = []
            for idx, benchmark in enumerate(benchmarks):
                try:
                    benchmark["generated_response"] = response_function(
                        benchmark["query"]
                    )
                except Exception as e:
                    return {
                        "success": False,
                        "message": str(e),
                        "type": "User response function",
                    }
                evaluation_url = (
                    f"{self.LH.base_url}/api/{test_id}/docqa_evaluate_single"
                )
                evaluation_data = benchmark
                evaluation_headers = {
                    "api-key": self.LH.lh_api_key,
                }
                evaluation_response = requests.post(
                    evaluation_url, headers=evaluation_headers, json=evaluation_data
                )
                if evaluation_response.status_code == 200:
                    evaluation = evaluation_response.json()
                    results.append(evaluation)
                    print(f"Evaluated on benchmark {idx+1}/{len(benchmarks)}")
                else:
                    print("error")
                    return {
                        "success": False,
                        "message": evaluation_response.json()["message"],
                    }
            return {
                "success": True,
                "evaluation": results,
                "test_id": test_id,
                "benchmark_id": benchmark_id,
                "dashboard_url": f"https://lighthouz.ai/evaluation/{self.LH.lh_app_id}/{test_id}?api_key={self.LH.lh_api_key}",
            }

        else:
            return {
                "success": False,
                "message": test_create_response.json()["message"],
            }
