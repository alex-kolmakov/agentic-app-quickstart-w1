"""
Migrated Multi-Agent System with MCP Tools
Preserves all the sophisticated agent logic while using MCP for data operations
"""

# Try to import without triggering the problematic scripts module
import os
import sys

# Temporarily suppress tensorflow warnings/errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from agents import Agent, Runner, set_tracing_disabled, SQLiteSession
    from agents.mcp import MCPServerStreamableHttp
    from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to minimal implementation if agents package has issues
    raise ImportError("Could not import agents package - check dependencies")

from textwrap import dedent
from openai import OpenAI
from typing import List, Dict, Any, Optional
import json
import polars as pl
import logging

from .utils import get_mcp_server
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.model import get_model

set_tracing_disabled(True)

# Data Loader Agent - Uses MCP tools for file operations
data_loader_agent = Agent(
    name="DataLoaderAgent",
    instructions="""You are a Data Loader Agent specialized in CSV file operations and data preparation.

Your responsibilities:
📁 File Operations:
- Investigating directories to find appropriate data files (use investigate_directory tool)
- Loading CSV files from various paths (use load_csv_file tool)
- Validating file formats and structure
- Handling file errors and providing clear feedback
- Inspecting dataset metadata (use get_dataset_info tool)

🔍 Data Inspection:
- Providing dataset summaries and overviews
- Listing column names and data types (use get_column_names tool)
- Showing sample data for user understanding
- Identifying potential data quality issues

🎯 Smart File Selection:
- When users ask for specific types of data (e.g., "weather data", "sales data"), 
  first use investigate_directory() to find relevant files
- Analyze filenames and content hints to recommend the best file to load
- Provide clear explanations for file selection decisions

When working with users:
- Always verify the file path and existence before loading
- If unsure which file to load, investigate the directory first
- Provide clear success/error messages
- Give helpful context about the loaded dataset
- Hand off to Analytics Agent for calculations and analysis
- Hand off to Visualization Agent for chart creation
- Hand off to Communication Agent for final user presentation

You focus on the "input" side - getting data ready for analysis.
Once data is loaded and validated, transfer to appropriate specialists.""",
    model=get_model()
    # MCP tools will be added via mcp_servers
)

# Analytics Agent - Uses MCP tools for calculations
analytics_agent = Agent(
    name="AnalyticsAgent", 
    instructions="""You are an Analytics Agent specialized in statistical analysis and data calculations.

Your expertise includes:
📊 Statistical Analysis:
- Calculating averages, medians, and comprehensive statistics (use calculate_column_stats tool)
- Finding minimum and maximum values with context
- Performing aggregations and grouping operations
- Filtering and conditional analysis

🔍 Data Exploration:
- Identifying patterns and trends
- Comparing values across groups
- Finding outliers or unusual data points
- Correlation and relationship analysis

💡 Insights Generation:
- Interpreting statistical results
- Identifying significant findings
- Suggesting additional analysis opportunities
- Providing business context for numbers

When performing analysis:
- Ensure data is properly loaded before calculations
- Handle non-numeric data gracefully
- Provide context with your calculations (not just numbers)
- Hand off to Communication Agent for user-friendly presentation
- Suggest related analyses that might be valuable

You focus on the "processing" side - turning raw data into meaningful insights.""",
    model=get_model()
)

# Visualization Agent - Uses MCP tools for chart creation
visualization_agent = Agent(
    name="VisualizationAgent",
    instructions="""You are a Visualization Agent specialized in creating insightful charts and visual representations of data.

Your expertise includes:
📈 Chart Creation:
- Bar charts for categorical comparisons (use create_bar_chart tool)
- Scatter plots for relationship exploration (use create_scatter_plot tool)
- Box plots for outlier detection and distribution (use create_box_plot tool)

🎨 Visual Design:
- Choosing appropriate chart types for data
- Creating clear, readable visualizations
- Adding context and annotations to charts
- Ensuring professional appearance and styling

📊 Insight Generation:
- Identifying visual patterns and trends
- Highlighting important findings in charts
- Suggesting follow-up visualizations
- Explaining what the charts reveal about the data

When creating visualizations:
- Always ensure data is loaded and valid first
- Choose the most appropriate chart type for the question
- Export charts as high-quality images
- Provide clear descriptions of what the visualization shows
- Hand off to Communication Agent for user-friendly explanations
- Suggest additional visualizations that might be valuable

You focus on making data insights visual and accessible!""",
    model=get_model()
)

# Communication Agent - Formats responses (no MCP tools needed)
communication_agent = Agent(
    name="CommunicationAgent",
    instructions="""You are a Communication Agent specialized in making data insights accessible and engaging.

Your role is to:
💬 User-Friendly Communication:
- Translate technical results into plain English
- Provide context and interpretation for findings
- Use appropriate emojis and formatting for readability
- Adapt communication style to user's expertise level

📖 Educational Support:
- Explain statistical concepts when relevant
- Provide insights beyond just numbers
- Suggest practical implications of findings
- Guide users toward next steps or follow-up questions

🎯 Response Crafting:
- Structure responses logically and clearly
- Highlight key insights and important findings
- Ask clarifying questions when user intent is unclear
- Offer suggestions for additional analysis or visualizations

✨ User Experience:
- Maintain friendly and helpful tone
- Acknowledge user questions fully before responding
- Provide actionable insights and recommendations
- Build conversation flow and context

📈 Visualization Integration:
- Explain what charts and graphs reveal
- Highlight key patterns and insights from visualizations
- Suggest when visual analysis might be helpful
- Make chart findings accessible to all users

You receive analyzed data and visualizations from other agents and present them in the most helpful way.
You're the "output" specialist - making sure users understand and can act on insights.

Remember: Your goal is to make data analysis feel approachable, valuable, and visually engaging!""",
    model=get_model()
)

# Set up handoffs between agents (preserve original multi-agent orchestration)
data_loader_agent.handoffs = [analytics_agent, visualization_agent, communication_agent]
analytics_agent.handoffs = [data_loader_agent, visualization_agent, communication_agent]
visualization_agent.handoffs = [data_loader_agent, analytics_agent, communication_agent]
communication_agent.handoffs = [data_loader_agent, analytics_agent, visualization_agent]


async def run_multi_agent_analysis(question: str, session_id: str = "default"):
    """
    Run the multi-agent data analysis system with MCP tools
    
    Args:
        question (str): User's data analysis question
        session_id (str): Session identifier for conversation history
        
    Returns:
        str: Final response from the multi-agent system
    """
    
    # List of all MCP tools that were migrated from tools.py
    allowed_tool_names = [
        "investigate_directory",
        "load_csv_file", 
        "get_column_names",
        "get_dataset_info",
        "calculate_column_average",
        "calculate_column_stats",
        "count_rows_with_value",
        "get_unique_values",
        "find_max_value",
        "find_min_value",
        "group_by_column_and_aggregate",
        "filter_data",
        "create_bar_chart",
        "create_scatter_plot", 
        "create_box_plot"
    ]
    
    async with get_mcp_server(allowed_tool_names=allowed_tool_names) as mcp_server:
        
        # Assign MCP server to agents that need tools
        data_loader_agent.mcp_servers = [mcp_server]
        analytics_agent.mcp_servers = [mcp_server] 
        visualization_agent.mcp_servers = [mcp_server]
        # communication_agent doesn't need direct data tools
        
        
        # Run the multi-agent system
        response = await Runner.run(
            starting_agent=data_loader_agent, 
            input=question,
            session=SQLiteSession(session_id)
        )
        
        return response.final_output


# Keep the agent registry for compatibility
multi_agents = {
    "data_loader": data_loader_agent,
    "analytics": analytics_agent,
    "visualization": visualization_agent,
    "communication": communication_agent
}
