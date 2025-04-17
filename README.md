# Contextual Query Reformulation and Evaluation with Azure OpenAI

This repository provides an end-to-end pipeline for **generating** and **evaluating** context-aware user queries using **Azure OpenAI's GPT model**. It is designed to:
- Reformulate user queries into self-contained versions using chat history.
- Evaluate the reformulated queries against reference queries for semantic intent similarity.

---

## Project Structure

```
ContextualQuery_Evaluation/
│
├── data/
│   └── dataset.json      # Input dataset with chat_history, current_query, and expected_contextual_query
│
├── results_data/
│   ├── predicted_queries.json     # Output of generated contextual queries
│   ├── evaluation_results.json    # Evaluation report (match % and full breakdown)
│   ├── matched_results.json       # Matched results based on intent
│   └── mismatched_results.json    # Mismatched results based on intent
│
├── generate_queries.py   # Code for generating contextual queries
├── evaluate_queries.py   # Code for evaluating predicted queries
└── .env                  # Azure OpenAI credentials and config
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ContextualQuery_Evaluation.git
cd ContextualQuery_Evaluation
```

### 2. Install Dependencies

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate
```

Install required packages:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
python-dotenv
tqdm
langchain
langchain-openai
```

### 3. Configure `.env`

Create a `.env` file in the project root with the following keys:

```env
AZURE_USER_ID=your_user_id
AZURE_OPENAI_API_KEY=your_api_key
AZURE_DEPLOYMENT_NAME=your_deployment_name
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_API_VERSION=2023-05-15
```

## Run Contextual Query Generation

```bash
python generate_queries.py
```

This will:
* Read the input file from `data/dataset.json`
* Use Azure OpenAI to generate contextual queries
* Save the output to `results_data/predicted_queries.json`

## Run Query Evaluation

```bash
python evaluate_queries.py
```

This will:
* Compare predicted vs. expected queries for semantic similarity
* Save the evaluation report and split results into matched/mismatched files

## Output Metrics

After evaluation, check:
* `evaluation_results.json`: Overall match % and reasoning
* `matched_results.json`: Successfully matched intents
* `mismatched_results.json`: Intents that were considered different

## Example Dataset Format

```json
{
  "chat_history": [
    {"role": "user", "content": "Tell me about your premium plans."},
    {"role": "assistant", "content": "We offer monthly and yearly plans with additional features."}
  ],
  "current_query": "Can I get a discount?",
  "contextual_query": "Can I get a discount on your premium plans?"
}
```
