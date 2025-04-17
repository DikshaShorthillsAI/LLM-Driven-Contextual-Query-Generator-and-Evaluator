import json
import os
import time
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FILE_PATH = "/home/shtlp_0010/Desktop/ContextualQuery_Evaluation/data/dataset.json"
QUERY_OUTPUT_PATH = "/home/shtlp_0010/Desktop/ContextualQuery_Evaluation/results_data/predicted_queries.json"
OUTPUT_DIR = "/home/shtlp_0010/Desktop/ContextualQuery_Evaluation/results_data"

BATCH_SIZE = 5
SLEEP_BETWEEN_BATCHES = 2

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

        # Initialize the LLM model
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

chat_model = CustomAzureChatOpenAI(max_tokens=100, temperature=0)

def generate_response(prompt):
    """Generate response using Azure OpenAI."""
    try:
        return chat_model.chat(prompt).strip()
    except Exception as e:
        logging.error(f"Azure OpenAI error: {e}")
        return ""

def generate_contextual_queries(dataset):
    """Generates and saves contextual queries for evaluation."""
    results = []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing Queries", unit="batch"):
        batch = dataset[i:i + BATCH_SIZE]
        for data in batch:
            chat_history = data["chat_history"]
            current_query = data["current_query"]
            expected_query = data["contextual_query"]

            reformulation_prompt = f'''
            You are an **expert in query reformulation**. Convert the user's query into a self-contained version using chat history while preserving intent.
            
            ### **Input:**
            - **Chat History:** {json.dumps(chat_history, indent=2)}
            - **User Query:** "{current_query}"
            
            ### **Output:**
            Return only the **reformulated query**, nothing else.
            '''

            predicted_query = generate_response(reformulation_prompt)
            if not predicted_query:
                continue  

            results.append({
                "chat_history": chat_history,
                "current_query": current_query,
                "expected_contextual_query": expected_query,
                "predicted_contextual_query": predicted_query
            })
        
        logging.info("Sleeping for rate limit prevention...")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(QUERY_OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved generated queries to {QUERY_OUTPUT_PATH}")

if __name__ == "__main__":
    with open(FILE_PATH, "r") as f:
        dataset = json.load(f)
    generate_contextual_queries(dataset)
