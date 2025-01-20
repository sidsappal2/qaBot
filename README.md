# QA Chat Bot with Caching and VectorStore

## Overview

This repository provides a QA (Question-Answering) chatbot application built with Streamlit. The application leverages cutting-edge technologies, such as language models, vector stores, and SQLite caching, to deliver fast and accurate responses to user queries. It supports dynamic data processing from user-provided URLs and maintains a history of interactions.

---

## Features

- **Dynamic Document Loading**: Processes user-provided URLs to build a searchable knowledge base.
- **Language Model Integration**: Utilizes ChatGroq and HuggingFace models for natural language understanding.
- **Persistent VectorStore**: Powered by ChromaDB for storing embeddings and metadata.
- **Caching with SQLite**: Reduces redundant computations by storing and retrieving answers efficiently.
- **Interaction History**: Maintains a detailed history of questions, answers, and sources.
- **Streamlit UI**: Easy-to-use interface for asking questions, processing URLs, and viewing history.

---

## File Descriptions

### `main.py`
- **Purpose**: The primary script that initializes and runs the Streamlit app.
- **Key Functionalities**:
  - Loads environment variables and sets up logging.
  - Initializes components: VectorStore, language models (LLM), and cache.
  - Handles user interactions, such as processing URLs, asking questions, and viewing history.
  - Implements caching and history management for enhanced efficiency.
- **Dependencies**: Streamlit, LangChain, SQLite3, threading, and dotenv.

### `createvector.py`
- **Purpose**: Manages the persistent vector store using ChromaDB.
- **Key Functionalities**:
  - Initializes a vector store collection.
  - Loads or updates the vector store with new document embeddings.
  - Handles vectorization and metadata management.
- **Technologies**: ChromaDB, LangChain, UUID.

### `utils.py`
- **Purpose**: Provides utility functions for cleaning and preprocessing text.
- **Key Functionalities**:
  - Removes HTML tags, URLs, special characters, and extra spaces.
  - Converts text to uppercase for consistent processing.
- **Use Case**: Ensures input consistency and prevents redundant calls to the language model.

---
# README: QA Bot Evaluation Framework

This repository contains an evaluation framework for assessing the performance of a Question-Answering (QA) bot. The framework measures metrics related to retrieval accuracy, answer quality, and response latency. Additionally, it provides tools for visualizing these metrics.

---

## Table of Contents

1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Functions](#functions)
4. [Usage](#usage)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Visualization](#visualization)
7. [Example Test Data](#example-test-data)

---

## Overview

This framework evaluates QA bots on the following aspects:

1. **Retrieval Metrics**: Precision, Recall, and F1-score of retrieved sources compared to relevant sources.
2. **Answer Quality**: BLEU and ROUGE scores for comparing the bot's generated answers with the expected answers.
3. **Response Latency**: The time taken by the bot to respond to a query.

---

## Dependencies

Ensure the following Python libraries are installed before running the framework:

- `sklearn` for calculating precision, recall, and F1-score
- `nltk` for BLEU score evaluation
- `rouge_score` for ROUGE evaluation
- `matplotlib` for plotting evaluation metrics

Install the required libraries using pip:

```bash
pip install scikit-learn nltk rouge_score matplotlib
```

---

## Functions

### 1. `evaluate_retrieval(true_sources, predicted_sources)`
Calculates precision, recall, and F1-score for the retrieval component.
- **Inputs**: 
  - `true_sources`: List of ground truth sources.
  - `predicted_sources`: List of sources predicted by the bot.
- **Outputs**: Precision, Recall, F1-score.

### 2. `evaluate_answer(expected_answer, generated_answer)`
Calculates BLEU and ROUGE scores for evaluating the generated answers.
- **Inputs**:
  - `expected_answer`: The ground truth answer.
  - `generated_answer`: The answer generated by the bot.
- **Outputs**: BLEU score, ROUGE scores.

### 3. `measure_latency(func, ques)`
Measures the time taken by a function to respond to a query.
- **Inputs**:
  - `func`: The function being tested.
  - `ques`: The query or question.
- **Outputs**: Latency, function result.

### 4. `evaluate_bot(test_data)`
Main function to evaluate the bot on a dataset of test cases.
- **Inputs**:
  - `test_data`: A list of dictionaries containing `question`, `expected_answer`, and `relevant_sources`.
- **Outputs**: Latency metrics, retrieval metrics.

### 5. `plot_metrics(latencies, retrieval_metrics)`
Generates visualizations for evaluation metrics:
- Response latency histogram.
- Retrieval metrics (Precision, Recall, F1-score) over test cases.



## Evaluation Metrics

### Retrieval Metrics:
- **Precision**: Proportion of relevant sources in the retrieved sources.
- **Recall**: Proportion of relevant sources retrieved.
- **F1-Score**: Harmonic mean of precision and recall.

### Answer Quality:
- **BLEU Score**: Measures n-gram overlap between the expected and generated answers.
- **ROUGE Scores**: Measures overlap of unigrams, bigrams, and longest common subsequences.

### Response Latency:
- Measures the time (in seconds) taken by the bot to respond to a query.

---

## Visualization

The `plot_metrics` function generates two types of plots:

1. **Response Latencies**:
   - Histogram showing the distribution of latencies.

2. **Retrieval Metrics**:
   - Line plot showing Precision, Recall, and F1-score for each test case.

---

## Example Test Data

```python
example_test_data = [
    {
        "question": "What is the capital of France?",
        "expected_answer": "Paris",
        "relevant_sources": ["source1"]
    },
    {
        "question": "Who wrote '1984'?",
        "expected_answer": "George Orwell",
        "relevant_sources": ["source2", "source3"]
    }
]

evaluate_bot(example_test_data)
```

---

## Note
- This framework assumes that your QA bot integrates seamlessly with the `ask_question` function. Modify this function as per your bot's API or structure.
- For large test datasets, consider optimizing the evaluation process to handle latency and resource constraints effectively.

---
