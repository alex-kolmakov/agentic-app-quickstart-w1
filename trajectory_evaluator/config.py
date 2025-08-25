"""
Configuration and Environment Setup for Trajectory Evaluator

This module handles configuration loading and environment variable management.
Uses the main project's .env file for configuration.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv


class Config:
    """Configuration management for the trajectory evaluator."""
    
    def __init__(self):
        # Load environment variables from project root
        project_root = Path(__file__).parent.parent
        env_file = project_root / ".env"
        load_dotenv(env_file)
        
        # Phoenix configuration (local container)
        self.phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://phoenix:6006")
        self.phoenix_project_name = os.getenv("PHOENIX_PROJECT_NAME", "agentic-app-quickstart")
        
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_endpoint = os.getenv("OPENAI_API_ENDPOINT", "https://api.openai.com/v1")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Evaluation settings
        self.default_lookback_days = int(os.getenv("DEFAULT_LOOKBACK_DAYS", "1"))
        self.default_evaluation_types = os.getenv(
            "DEFAULT_EVALUATION_TYPES", 
            "duplicate_tools,handoff_loops,efficiency"
        ).split(",")
        
        # Output configuration
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        self.output_dir.mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.openai_api_key:
            print("❌ OPENAI_API_KEY is required")
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "phoenix_endpoint": self.phoenix_endpoint,
            "phoenix_project_name": self.phoenix_project_name,
            "openai_model": self.openai_model,
            "default_lookback_days": self.default_lookback_days,
            "default_evaluation_types": self.default_evaluation_types,
            "output_dir": str(self.output_dir)
        }
    
    def print_config(self) -> None:
        """Print current configuration (excluding sensitive data)."""
        print("🔧 Configuration:")
        print(f"   Phoenix Endpoint: {self.phoenix_endpoint}")
        print(f"   Phoenix Project: {self.phoenix_project_name}")
        print(f"   OpenAI Model: {self.openai_model}")
        print(f"   Output Directory: {self.output_dir}")
        print(f"   Default Lookback: {self.default_lookback_days} days")
        print(f"   Default Evaluations: {', '.join(self.default_evaluation_types)}")


# Global configuration instance
config = Config()
