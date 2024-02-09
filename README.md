# ![lighthouz](https://lighthouz.ai/lighthouz-logo.png)

<div align="center">

![PyPI - Version](https://img.shields.io/pypi/v/lighthouz?label=lighthouz&link=https%3A%2F%2Fpypi.org%2Fproject%2Flighthouz)
[![Docs](https://img.shields.io/badge/docs-lighthouz%20docs-green)](https://www.lighthouz.ai/docs/)
[![GitHub](https://img.shields.io/badge/github-Lighthouz_AI-blue)](https://github.com/Lighthouz-AI)

</div>

Lighthouz AI is a premier AI benchmarking, evaluation, and security platform. It is meticulously designed to aid
developers in both evaluating the reliability and enhancing the capabilities of their Language Learning Model (LLM)
applications.

## Installation

```bash
pip install lighthouz
```

if you want to install from source

```bash
pip install git+https://github.com/Lighthouz-AI/lighthouz_sdk
```

## Quick Usage

### Initialization

```python
from lighthouz import Lighthouz

LH = Lighthouz("lighthouz_api_key")  # replace with your lighthouz api key
```

### AutoBench

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

### Evaluate a RAG Application

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

## Documentation

Please refer to the [SDK docs](https://lighthouz.ai/docs/) for how to use.
