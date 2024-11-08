import openai
import os
import time
from functools import wraps
from azure.identity import ClientSecretCredential
from langchain.chat_models import AzureChatOpenAI


AZ_OAI_API_BASE_GPT_3 = os.getenv("AZ_OAI_API_BASE_GPT_3")
AZ_OAI_API_BASE_GPT_4 = os.getenv("AZ_OAI_API_BASE_GPT_4")
AZ_OAI_API_BASE_GPT_4_32_k = os.getenv("AZ_OAI_API_BASE_GPT_4_32_k")
AZ_OAI_DEPLOYMENT_ID_GPT_3 = os.getenv("AZ_OAI_DEPLOYMENT_ID_GPT_3")
AZ_OAI_DEPLOYMENT_ID_GPT_4 = os.getenv("AZ_OAI_DEPLOYMENT_ID_GPT_4")
AZ_OAI_DEPLOYMENT_ID_GPT_4_32_k = os.getenv("AZ_OAI_DEPLOYMENT_ID_GPT_4_32_k")
AZ_OAI_DEPLOYMENT_ID_ADA_2 = os.getenv("AZ_OAI_DEPLOYMENT_ID_ADA_2")


openai.api_type = "azuread"
openai.api_base = AZ_OAI_API_BASE_GPT_3
openai.api_version = "2023-05-15"

os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = AZ_OAI_API_BASE_GPT_3

# Global variables to store the token and its expiration time
# Global variables to store the token and its expiration time
current_token = None
token_expiration_time = None


def get_token():
    global current_token, token_expiration_time

    # Check if the token is about to expire (or has already expired)
    if current_token is None or token_expiration_time is None or (token_expiration_time - time.time()) < 60:
        credential = ClientSecretCredential(
            os.getenv("AZ_TENANT_ID"),
            os.getenv("AZ_SP_CLIENT_ID"),
            os.getenv("AZ_SP_CLIENT_SECRET"),
        )
        current_token = credential.get_token("https://cognitiveservices.azure.com/.default")

        # Update the expiration time
        # Assuming the token lasts for 1 hour, adjust as necessary
        token_expiration_time = time.time() + 3600

    return current_token


def retry(max_retries=5, delay=3):
    """A decorator to retry function execution in case of errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f'Error: {e}, retrying...')
                    time.sleep(delay)
            return func(*args, **kwargs)
        return wrapper
    return decorator


@retry(max_retries=5, delay=3)
def get_embedding(doc, model="gpt-3.5-turbo"):
    openai.api_base = AZ_OAI_API_BASE_GPT_3
    openai.api_key = get_token().token

    response = openai.Embedding.create(
        input=doc,
        # model=model,
        deployment_id=AZ_OAI_DEPLOYMENT_ID_ADA_2
    )
    return response['data'][0]['embedding']


@retry(max_retries=5, delay=3)
def get_completion_4(prompt, model="gpt-4"):
    openai.api_base = AZ_OAI_API_BASE_GPT_4
    openai.api_key = get_token().token
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        deployment_id=AZ_OAI_DEPLOYMENT_ID_GPT_4
    )
    return response.choices[0].message["content"]


@retry(max_retries=1, delay=3)
def get_completion_4_32(prompt):
    openai.api_base = AZ_OAI_API_BASE_GPT_4_32_k
    openai.api_key = get_token().token
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        messages=messages,
        temperature=0,
        deployment_id=AZ_OAI_DEPLOYMENT_ID_GPT_4_32_k
    )
    return response.choices[0].message["content"]


def get_langchain_llm_4():
    openai.api_base = AZ_OAI_API_BASE_GPT_4
    os.environ["OPENAI_API_BASE"] = AZ_OAI_API_BASE_GPT_4
    os.environ["OPENAI_API_KEY"] = get_token().token
    return AzureChatOpenAI(
        openai_api_base=AZ_OAI_API_BASE_GPT_4,
        openai_api_version="2023-05-15",
        deployment_name=AZ_OAI_DEPLOYMENT_ID_GPT_4,
        openai_api_type="azuread",
    )


def get_langchain_llm_4_32_k():
    openai.api_base = AZ_OAI_API_BASE_GPT_4_32_k
    os.environ["OPENAI_API_BASE"] = AZ_OAI_API_BASE_GPT_4_32_k
    os.environ["OPENAI_API_KEY"] = get_token().token
    return AzureChatOpenAI(
        openai_api_base=AZ_OAI_API_BASE_GPT_4_32_k,
        openai_api_version="2023-05-15",
        deployment_name=AZ_OAI_DEPLOYMENT_ID_GPT_4_32_k,
        openai_api_type="azuread",
    )


