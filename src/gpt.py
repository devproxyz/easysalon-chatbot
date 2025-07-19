from openai import OpenAI

def get_gpt_client(url, key):
    client = OpenAI(
        api_key=key,  # Retrieves API key from .env
        base_url=url,  # Retrieves endpoint URL
    )
    return client