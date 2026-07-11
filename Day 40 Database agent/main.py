import os
from mistralai import Mistral
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()
# Initialize Mistral API client with your API key
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


@traceable(
    run_type="llm",
    metadata={"ls_provider": "mistral", "ls_model_name": "mistral-medium-latest"},
)
def query_mistral(prompt: str):
    response = client.chat.complete(
        model="mistral-medium-latest",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message


# Example usage
result = query_mistral("Hello, how are you?")
print("Mistral response:", result.content)