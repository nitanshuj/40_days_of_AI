import os
from dotenv import load_dotenv
from perplexity import Perplexity

load_dotenv()


api_key = os.getenv("PERPLEXITY_API_KEY")
print("API key found!" if api_key else "API key not found.")

# API key is detected from environment variables
client = Perplexity()

# Simple search request
search = client.search.create(
    query="latest AI developments 2024",
    max_results=5,
    max_tokens_per_page=1024
)

for result in search.results:
    print(f"{result.title}: {result.url}")