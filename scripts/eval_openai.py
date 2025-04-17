import json
import os
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QUERY_OUTPUT_PATH = "/home/shtlp_0010/Desktop/ContextualQuery_Evaluation/results_data/predicted_queries.json"
OUTPUT_DIR = "/home/shtlp_0010/Desktop/ContextualQuery_Evaluation/results_data"
EVAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "evaluation_results.json")
MATCHED_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matched_results.json")
MISMATCHED_RESULTS_PATH = os.path.join(OUTPUT_DIR, "mismatched_results.json")

class CustomAzureChatOpenAI:
    def __init__(self, max_tokens=100, temperature=0):
        """
        Initializes the AzureChatOpenAI instance with credentials and parameters.
        """
        self.user_id = os.getenv("AZURE_USER_ID")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_API_VERSION")

        self.llm = AzureChatOpenAI(
            default_headers={"User-Id": self.user_id},
            temperature=temperature,
            azure_deployment=self.deployment,
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            timeout=100,
            api_version=self.api_version,
            max_tokens=max_tokens,
        )

    def chat(self, message: str):
        """
        Sends a message to the Azure OpenAI chat model and returns the response.
        """
        response = self.llm.invoke([HumanMessage(message)])
        return response.content

def extract_json_response(response_text):
    """Safely extract JSON content from the response."""
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}")
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON found in response")
        json_data = response_text[json_start:json_end+1]
        return json.loads(json_data)
    except (json.JSONDecodeError, ValueError) as e:
        logging.error(f"Failed to parse LLM response: {response_text}")
        return {"match": False, "reasoning": f"Failed to parse response: {str(e)}"}

def evaluate_queries():
    """Evaluate generated contextual queries and save matched/mismatched results separately."""
    chat_model = CustomAzureChatOpenAI(max_tokens=150, temperature=0)
    
    with open(QUERY_OUTPUT_PATH, "r") as f:
        generated_queries = json.load(f)
    
    results = []
    matched_results = []
    mismatched_results = []
    total, true_matches = 0, 0
    
    for data in tqdm(generated_queries, desc="Evaluating Queries", unit="query"):
        chat_history = data["chat_history"]
        current_query = data["current_query"]
        expected_query = data["expected_contextual_query"]
        predicted_query = data["predicted_contextual_query"]

        eval_prompt = f'''
        You are an AI specializing in **query similarity assessment**. Determine whether the **expected query** and **predicted query** have the same intent within the given chat history.
        
        ### **Input:**
        - **Chat History:** {json.dumps(chat_history, indent=2)}
        - **User Query:** "{current_query}"
        - **Expected Query:** "{expected_query}"
        - **Predicted Query:** "{predicted_query}"
        
        ### **Output:**
        ```json
        {{
            "match": true or false,
            "reasoning": "<concise explanation>"
        }}
        ```
        '''

        response_text = chat_model.chat(eval_prompt)
        response_json = extract_json_response(response_text)
        match = response_json.get("match", False)
        reasoning = response_json.get("reasoning", "No explanation provided.")

        total += 1
        if match:
            true_matches += 1
            matched_results.append({
                "chat_history": chat_history,
                "current_query": current_query,
                "expected_contextual_query": expected_query,
                "predicted_contextual_query": predicted_query,
                "match": match,
                "reasoning": reasoning
            })
        else:
            mismatched_results.append({
                "chat_history": chat_history,
                "current_query": current_query,
                "expected_contextual_query": expected_query,
                "predicted_contextual_query": predicted_query,
                "match": match,
                "reasoning": reasoning
            })

        results.append({
            "chat_history": chat_history,
            "current_query": current_query,
            "expected_contextual_query": expected_query,
            "predicted_contextual_query": predicted_query,
            "match": match,
            "reasoning": reasoning
        })
    
    match_percentage = (true_matches / total) * 100 if total else 0
    false_percentage = 100 - match_percentage if total else 0
    logging.info(f"Match Percentage: {match_percentage:.2f}% | False Percentage: {false_percentage:.2f}%")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(EVAL_OUTPUT_PATH, "w") as f:
        json.dump({
            "match_percentage": match_percentage,
            "false_percentage": false_percentage,
            "results": results
        }, f, indent=4)
    
    with open(MATCHED_RESULTS_PATH, "w") as f:
        json.dump(matched_results, f, indent=4)
    
    with open(MISMATCHED_RESULTS_PATH, "w") as f:
        json.dump(mismatched_results, f, indent=4)

    logging.info(f"Saved evaluation results to {EVAL_OUTPUT_PATH}")
    logging.info(f"Saved matched results to {MATCHED_RESULTS_PATH}")
    logging.info(f"Saved mismatched results to {MISMATCHED_RESULTS_PATH}")

if __name__ == "__main__":
    evaluate_queries()
