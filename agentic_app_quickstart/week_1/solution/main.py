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

# Add the parent directory to the path so we can import from the package
sys.path.append(str(Path(__file__).parent.parent.parent))

from agentic_app_quickstart.week_1.solution.custom_agents import (
    run_multi_agent_conversation,
    run_demo
)


def print_welcome():
    """Print the welcome message and system overview."""
    print("🎪📈" + "=" * 58 + "🎪📈")
    print("    MULTI-AGENT CSV DATA ANALYSIS & VISUALIZATION SYSTEM")
    print("🎪📈" + "=" * 58 + "🎪📈")
    print()
    print("🚀 Welcome to your intelligent data analysis & visualization assistant!")
    print()
    print("💡 EXAMPLE QUESTIONS:")
    print('   Data Loading:')
    print('   "Load the employee_data.csv file"')
    print('   "What are the column names?"')
    print()
    print('   Statistical Analysis:')
    print('   "What\'s the average salary?"')
    print('   "How many employees are in Engineering?"')
    print('   "Who has the highest performance score?"')
    print()
    print('   Visualization Requests:')
    print('   "Create a dashboard for the employee data"')
    print('   "Show me a histogram of salaries"')
    print('   "Make a scatter plot of salary vs performance"')
    print('   "Create a bar chart showing average salary by department"')
    print('   "Show a box plot of performance scores by department"')
    print('   "Generate a correlation heatmap"')
    print()
    print("🎪📈" + "=" * 58 + "🎪📈")


async def main():
    """Main function to run the multi-agent data analysis & visualization system."""
    print_welcome()

    while True:
        await run_multi_agent_conversation()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the Multi-Agent Data Analysis & Visualization System!")
        print("🎨 Don't forget to check your directory for any generated charts!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("Please check your setup and try again.")
