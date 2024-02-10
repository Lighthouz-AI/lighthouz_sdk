<div align="center">
  <img src="https://lighthouz.ai/lighthouz-logo.png" alt="lighthouz" width="50%"/>
</div>

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/lighthouz?label=lighthouz&link=https%3A%2F%2Fpypi.org%2Fproject%2Flighthouz)
[![Docs](https://img.shields.io/badge/docs-lighthouz%20docs-green)](https://www.lighthouz.ai/docs/)
[![GitHub](https://img.shields.io/badge/github-Lighthouz_AI-blue)](https://github.com/Lighthouz-AI)

[Installation](#installation) | [Quick Usage](#quick-usage) | [Documentation](https://www.lighthouz.ai/docs/)

</div>

Lighthouz AI is a AI benchmark data generation, evaluation, and security platform. It is meticulously designed to aid
developers in both evaluating the reliability and enhancing the capabilities of their Language Learning Model (LLM)
applications.


## Key Features

Lighthouz has the following features: 

### 1. AutoBench: Create custom benchmarks 
- **Create Benchmarks**: AutoBench creates application-specific and task-specific benchmark test cases to assess critical reliability, security, and privacy aspects of your LLM app. 
- **Flexibility**: Tailor-made benchmarks to suit your specific evaluation needs.
- **Integration with your own benchmarks**: Seamlessly upload and incorporate your pre-existing benchmarks.

### 2. Eval Studio: Evaluate LLM Applications
- **Comprehensive Analysis**: Thoroughly assess your LLM application for hallucinations, toxicity, out-of-context responses, PII data leaks, and prompt injections.
- **Insightful Feedback**: Gain valuable insights to refine your application.
- **Comparative Analysis**: Effortlessly compare different LLM apps and versions.
- **Customization**: Test the impact on performance of prompts, LLMs, hyperparameters, etc.

### 3. Watchtower: Monitoring and Security
- **Real-Time Monitoring**: Keep a vigilant eye on your LLM applications.
- **Metrics**: View detailed metrics spanning hallucinations, quality, prompt injection, PII leaks, and more.
- **Enhanced Security**: Proactive measures to safeguard your LLM app against vulnerabilities like hallucination, data leak, prompt injection, and more.
- **Easy API integration**: easily call Lighthouz API to log and evaluate all calls.


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


### Evaluate a RAG Application on a Benchmark Dataset 

Shows how to use the Evaluation class from Lighthouz to evaluate a RAG system. It involves initializing an
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


### Use Lighthouz Eval Endpoint to Evaluate a Single RAG Query

Add your Lighthouz API key before running the following code: 

```
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

## Quick Start Examples 

[Evaluation of a RAG app built with LangChain](https://lighthouz.ai/docs/examples/langchain-example)

[Evaluation of a RAG app built with LlamaIndex](https://lighthouz.ai/docs/examples/llamaindex-example)

[Evaluation of a RAG app hosted on an API endpoint](https://lighthouz.ai/docs/examples/api-example)

## Contact

For any queries, reach out to contact@lighthouz.ai

