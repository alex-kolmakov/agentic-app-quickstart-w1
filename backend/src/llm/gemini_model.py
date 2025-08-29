from openai import AsyncOpenAI
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
import os
from dotenv import load_dotenv

load_dotenv()

def get_gemini_client():
    """Get Gemini client for judge evaluation"""
    return AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )

def get_gemini_model():
    """Get Gemini model for judge evaluation"""
    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash",
        openai_client=get_gemini_client()
    )
    return model
