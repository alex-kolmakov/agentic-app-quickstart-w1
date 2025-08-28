from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv

load_dotenv()

def get_client():
    return AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def get_model():

    model = OpenAIChatCompletionsModel(
        model = "gemini-2.5-flash",
        openai_client=get_client()
    )

    return model