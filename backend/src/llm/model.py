import os
import openai
from typing import Optional


def get_model(model_name: Optional[str] = None):
    """Get OpenAI model configuration"""
    
    # Use environment variable or default
    model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4")
    
    # Configure OpenAI client
    openai_client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    return model_name  # Agents SDK will use this with the global OpenAI client
