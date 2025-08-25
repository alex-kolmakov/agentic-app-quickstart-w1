"""
Multi-Agent CSV Data Analysis System with Visualization - Main Entry Point

This is the main entry point for the enhanced CSV data analysis system with visualization capabilities.
It provides a specialized multi-agent approach for analyzing CSV data with natural language queries
and creating insightful visualizations.

🎯 Features:
- 📊 Load and analyze CSV files
- 🗣️ Natural language interface
- � Advanced data visualization (charts, plots, dashboards)
- �🔍 Statistical analysis and insights
- 💬 Memory-enabled conversations
- 🛡️ Robust error handling

🏗️ Multi-Agent Architecture:
- 📁 DataLoaderAgent: File operations and data preparation
- 📊 AnalyticsAgent: Statistical calculations and analysis
- 📈 VisualizationAgent: Chart creation and visual insights
- 💬 CommunicationAgent: User-friendly response formatting

🌟 Visualization Capabilities:
- 📈 Bar charts for categorical comparisons
- 🔍 Scatter plots for relationship exploration
- 📦 Box plots for outlier detection

Run this file to start the interactive visualization system!
"""

import asyncio
import sys
from pathlib import Path
import os
import gradio as gr

# Add the parent directory to the path so we can import from the package
sys.path.append(str(Path(__file__).parent.parent.parent))

from multi_agent_data_app.custom_agents import create_gradio_interface


async def main():
    """Main function to run the multi-agent data analysis & visualization system."""

    print("\n🌐 Starting Gradio Web Interface...")
    interface = create_gradio_interface()
    interface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the Multi-Agent Data Analysis & Visualization System!")
        print("🎨 Don't forget to check your directory for any generated charts!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("Please check your setup and try again.")
