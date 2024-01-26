import webbrowser
from typing import Callable

import requests

from lighthouz import Lighthouz


class Evaluation:
    def __init__(self, LH: Lighthouz):
        self.LH = LH

    def evaluate_rag_model(
            self, benchmark_id: str, app_id: str, response_function: Callable[[str], str]
    ):
        test_create_url = f"{self.LH.base_url}/apps/{app_id}/tests/create"
        test_create_data = {"status": "completed", "benchmark_id": benchmark_id}
        test_create_headers = {
            "api-key": self.LH.lh_api_key,
        }
        test_create_response = requests.post(
            test_create_url, headers=test_create_headers, json=test_create_data
        )
        if test_create_response.status_code == 200:
            test_id = test_create_response.json()["test_id"]
            benchmark_url = f"{self.LH.base_url}/benchmarks/{benchmark_id}"
            benchmark_headers = {
                "api-key": self.LH.lh_api_key,
            }
            benchmark_response = requests.get(benchmark_url, headers=benchmark_headers)
            if benchmark_response.status_code == 200:
                benchmarks = benchmark_response.json()["benchmark"]["benchmark"]
            else:
                return {
                    "success": False,
                    "message": benchmark_response.json().get("msg"),
                }

            print(f"Evaluating on {len(benchmarks)} test case(s).")
            results = []
            for idx, benchmark in enumerate(benchmarks):
                try:
                    generated_response = response_function(benchmark["query"])
                    if not isinstance(generated_response, str):
                        print(
                            {
                                "success": False,
                                "message": "The response function must return a string.",
                            }
                        )
                    benchmark["generated_response"] = generated_response
                except Exception as e:
                    print(
                        {
                            "success": False,
                            "message": str(e),
                            "type": "User response function error. Please check that you have provided a function "
                                    "that can take a query as an input.",
                        }
                    )
                    continue
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
                    print(f"Evaluated on test {idx + 1}/{len(benchmarks)}")
                else:
                    print(
                        "Error when evaluating test {idx + 1}/{len(benchmarks)}. Skipping."
                    )

            results_url = f"https://lighthouz.ai/evaluation/{app_id}/{test_id}?api_key={self.LH.lh_api_key}"
            print(
                f"Evaluation on benchmark {benchmark_id} is complete. The results can be viewed at ",
                results_url,
            )
            webbrowser.open(results_url)
            return {
                "success": True,
                # "evaluation": results,
                "test_id": test_id,
                "benchmark_id": benchmark_id,
                "dashboard_url": f"https://lighthouz.ai/evaluation/{app_id}/{test_id}?api_key={self.LH.lh_api_key}",
            }

        else:
            print("An error has occurred: ", test_create_response.json())
            return {
                "success": False,
                "message": test_create_response.json()["msg"],
            }

    def evaluate_multiple_rag_models(
            self,
            benchmark_id: str,
            app_ids: list[str],
            response_functions: list[Callable[[str], str]],
    ):
        if len(app_ids) != len(response_functions):
            return {
                "success": False,
                "message": "app_ids and response_functions must be of the same length",
            }
        evaluations = []
        for app_id, response_function in zip(app_ids, response_functions):
            print(f"Evaluating app {app_id} on the benchmark {benchmark_id}")
            evaluation = self.evaluate_rag_model(
                benchmark_id, app_id, response_function
            )
            evaluations.append(evaluation)
            print()

        return evaluations
