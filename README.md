<div align="center">
  <img src="https://lighthouz.ai/lighthouz-logo.png" alt="lighthouz" width="50%"/>
</div>

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/lighthouz?label=lighthouz&link=https%3A%2F%2Fpypi.org%2Fproject%2Flighthouz)
[![Docs](https://img.shields.io/badge/docs-lighthouz%20docs-green)](https://www.lighthouz.ai/docs/)
[![GitHub](https://img.shields.io/badge/github-Lighthouz_AI-blue)](https://github.com/Lighthouz-AI)

[Website](https://lighthouz.ai/) | [Installation](#installation) | [Quick Usage](#quick-usage) | [Documentation](https://www.lighthouz.ai/docs/)

</div>

Lighthouz AI is a AI benchmark data generation, evaluation, and security platform. It is meticulously designed to aid
developers in both evaluating the reliability and enhancing the capabilities of their Language Learning Model (LLM)
applications.

## Key Features

Lighthouz has the following features:

### 1. AutoBench: Create AI-assisted custom benchmarks for security, privacy, and reliability

- **Create Benchmarks**: AutoBench creates application-specific and task-specific benchmark test cases to assess
  critical security, privacy, and reliability aspects of your LLM app.
- **Flexibility**: Tailor-made benchmarks to suit your specific evaluation needs.
- **Integration with your own benchmarks**: Seamlessly upload and incorporate your pre-existing benchmarks.

### 2. Eval Studio: Evaluate LLM Applications for security, privacy, and reliability

- **Comprehensive Analysis**: Thoroughly assess your LLM application for hallucinations, toxicity, out-of-context
  responses, PII data leaks, and prompt injections.
- **Insightful Feedback**: Gain valuable insights to refine your application.
- **Comparative Analysis**: Effortlessly compare different LLM apps and versions.
- **Customization**: Test the impact on performance of prompts, LLMs, hyperparameters, etc.

## Installation

```bash
pip install lighthouz
```

## Quick Usage

### Initialization

```python
from lighthouz import Lighthouz

LH = Lighthouz("lighthouz_api_key")  # replace with your lighthouz api key
```

### AutoBench: Create custom benchmarks

To generate a benchmark, use the generate_benchmark function under the Benchmark class.

This generates and stores a benchmark spanning benchmark_category categories. The benchmark is a collection of unit
tests, called Prompt Unit Tests. Each unit test contains an input prompt, an expected response (if applicable),
context (if applicable), and corresponding file name (if applicable).

```python
from lighthouz.benchmark import Benchmark

lh_benchmark = Benchmark(LH)  # LH: Lighthouz instance initialized with Lighthouz API key
benchmark_data = lh_benchmark.generate_benchmark(
    file_path="pdf_file_path",
    benchmark_categories=["rag_benchmark", "out_of_context", "prompt_injection", "pii_leak"]
)
benchmark_id = benchmark_data.get("benchmark_id")
print(benchmark_id)
```

The possible `benchmark_categories` options are:

* "rag_benchmark": this creates two hallucination benchmarks, namely Hallucination: direct questions and Hallucination:
  indirect questions.
* "out_of_context": this benchmark contains out-of-context prompts to test whether the LLM app responds to irrelevant
  queries.
* "prompt_injection": this benchmark contains prompt injection prompts testing whether the LLM behavior can be
  manipulated.
* "pii_leak": this benchmark contains prompts testing whether the LLM can leak PII data.

The resulting data, when viewed on Lighthouz platform, looks as follows:
![AutoBench](https://lighthouz.ai/assets/images/autobench-ca4f6afca2405f37ce0de8f1e0c68f8e.png)

### Evaluate a RAG Application on a Benchmark Dataset

The following shows how to use the Evaluation class from Lighthouz to evaluate a RAG system. It involves initializing an
evaluation instance with a Lighthouz API key and using the evaluate_rag_model method with a response function, benchmark
ID, and app ID.

```python
from lighthouz.evaluation import Evaluation

evaluation = Evaluation(LH)  # LH: Lighthouz instance initialized with Lighthouz API key
e_single = evaluation.evaluate_rag_model(
    response_function=llamaindex_rag_query_function,
    benchmark_id="lighthouz_benchmark_id",  # replace with benchmark id
    app_id="lighthouz_app_id",  # replace with the app id
)
print(e_single)
```

The evaluation results, when viewed on Lighthouz platform, look as follows:
![AutoBench](https://lighthouz.ai/assets/images/eval-one-a075376733a726a70d0941e034f30a07.png)

Individual test cases are shown as:
![AutoBench](https://lighthouz.ai/assets/images/eval-one-detail-677b266dfb731953f826ef91aefe83b9.png)

### Use Lighthouz Eval Endpoint to Evaluate a Single RAG Query

Add your Lighthouz API key before running the following code:

```bash
curl -X POST "https://lighthouz.ai/api/api/evaluate_query" \
-H "api-key: YOUR LH API KEY" \
-H "Content-Type: application/json" \
-d '{
    "app_name": "gpt-4-0613",
    "query": "What is the Company'\''s line of personal computers based on its macOS operating system and what does it include?",
    "expected_response": "The Mac line includes laptops MacBook Air and MacBook Pro, as well as desktops iMac, Mac mini, Mac Studio and Mac Pro.",
    "generated_response": "The Company'\''s line of personal computers based on its macOS operating system is Mac.",
    "context": "s the Company’s line of smartphones based on its iOS operating system. The iPhone line includes iPhone 14 Pro, iPhone 14, iPhone 13, iPhone SE®, iPhone 12 and iPhone 11. Mac Mac® is the Company’s line of personal computers based on its macOS® operating system. The Mac line includes laptops MacBook Air® and MacBook Pro®, as well as desktops iMac®, Mac mini®, Mac Studio™ and Mac Pro®. iPad iPad® is the Company’s line of multipurpose tablets based on its iPadOS® operating system. The iPad line includes iPad Pro®, iPad Air®, iPad and iPad mini®. Wearables, Home and Accessories Wearables, Home and Accessories includes: •AirPods®, the Company’s wireless headphones, including AirPods, AirPods Pro® and AirPods Max™; •Apple TV®, the Company’s media streaming and gaming device based on its tvOS® operating system, including Apple TV 4K and Apple TV HD; •Apple Watch®, the Company’s line of smartwatches based on its watchOS® operating system, including Apple Watch Ultra ™, Apple Watch Series 8 and Apple Watch SE®; and •Beats® products, HomePod mini® and accessories. Apple Inc. | 2022 Form 10-K | 1"
}'
```

The returned result, in json format, is as follows:

```json
{
  "_id":"65c99a2a3ddb41f89115d327",
  "app_id":"65b6c0af56ecfafc9440b970",
  "app_title":"gpt-4-0613",
  "user_id":"658066787e7ab545580c0a98",
  "query":"What is the Company's line of personal computers based on its macOS operating system and what does it include?",
  "source_context":"s the Company\u2019s line of smartphones based on its iOS operating system. The iPhone line includes iPhone 14 Pro, iPhone 14, iPhone 13, iPhone SE\u00ae, iPhone 12 and iPhone 11. Mac Mac\u00ae is the Company\u2019s line of personal computers based on its macOS\u00ae operating system. The Mac line includes laptops MacBook Air\u00ae and MacBook Pro\u00ae, as well as desktops iMac\u00ae, Mac mini\u00ae, Mac Studio\u2122 and Mac Pro\u00ae. iPad iPad\u00ae is the Company\u2019s line of multipurpose tablets based on its iPadOS\u00ae operating system. The iPad line includes iPad Pro\u00ae, iPad Air\u00ae, iPad and iPad mini\u00ae. Wearables, Home and Accessories Wearables, Home and Accessories includes: \u2022AirPods\u00ae, the Company\u2019s wireless headphones, including AirPods, AirPods Pro\u00ae and AirPods Max\u2122; \u2022Apple TV\u00ae, the Company\u2019s media streaming and gaming device based on its tvOS\u00ae operating system, including Apple TV 4K and Apple TV HD; \u2022Apple Watch\u00ae, the Company\u2019s line of smartwatches based on its watchOS\u00ae operating system, including Apple Watch Ultra \u2122, Apple Watch Series 8 and Apple Watch SE\u00ae; and \u2022Beats\u00ae products, HomePod mini\u00ae and accessories. Apple Inc. | 2022 Form 10-K | 1",
  "expected_output":"The Mac line includes laptops MacBook Air and MacBook Pro, as well as desktops iMac, Mac mini, Mac Studio and Mac Pro.",
  "generated_output":"The Company's line of personal computers based on its macOS operating system is Mac.",
  "put_type":"Hallucination: Direct Question",
  "created_at":"Mon, 12 Feb 2024 04:10:18 GMT",
  "alerts":[],
  "label":"correct but incomplete",
  "passed":null,
  "similarity_score":0.6825323104858398,
  "conciseness_score":0.711864406779661,
  "query_toxicity_score":0.0008218016009777784,
  "generated_response_toxicity_score":0.0009125719661824405,
  "prompt_injection_score":0.00042253732681274414,
  "query_pii":[],
  "generated_response_pii":[],
}
```

## Quick Start Examples

[Evaluation of a RAG app built with LangChain](https://lighthouz.ai/docs/examples/langchain-example)

[Evaluation of a RAG app built with LlamaIndex](https://lighthouz.ai/docs/examples/llamaindex-example)

[Evaluation of a RAG app hosted on an API endpoint](https://lighthouz.ai/docs/examples/api-example)

## Contact

For any queries, reach out to contact@lighthouz.ai

