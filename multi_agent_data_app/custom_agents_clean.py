"""
Multi-Agent CSV Data Analysis System with MCP Integration

This module implements a specialized multi-agent approach for CSV data analysis with
MCP (Model Context Protocol) server integration for centralized tool management.

🎯 Features:
- 🤖🤖🤖 Multi-Agent System: Specialized agents for different aspects
- 📈 Data Visualization: Charts and plots via MCP server
- 💾 Memory: Conversation history and context preservation
- 🛡️ Error Handling: Robust error handling for user-friendly experience
- 🗣️ Natural Language Interface: Understands and responds to human questions
- 🔗 MCP Integration: All tools centralized in MCP server

Agent Specialists:
1. 📁 DataLoaderAgent - File operations and data preparation via MCP
2. 📊 AnalyticsAgent - Statistical calculations and analysis via MCP
3. 📈 VisualizationAgent - Chart creation and visual insights via MCP
4. 💬 CommunicationAgent - User-friendly response formatting
"""

import asyncio
from agents import Agent, Runner, SQLiteSession, set_tracing_disabled
from phoenix.otel import register
from multi_agent_data_app.helpers.helpers import get_model
from multi_agent_data_app.helpers.google_helpers import get_model as get_judge_model
from datetime import datetime
import gradio as gr
import os
from dotenv import load_dotenv
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().uninstrument()

load_dotenv()

# Enable Phoenix tracing for monitoring agent interactions
phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
phoenix_project_name = os.getenv("PHOENIX_PROJECT_NAME", "agentic-app-quickstart")

print(f"🔍 Initializing Phoenix tracing...")
print(f"   📡 Endpoint: {phoenix_endpoint}")
print(f"   📁 Project: {phoenix_project_name}")

tracer_provider = register(
    endpoint=phoenix_endpoint,
    project_name=phoenix_project_name,
)

# MCP Server Configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://mcp-server:8000")
print(f"🔗 MCP Server URL: {MCP_SERVER_URL}")

# =============================================================================
# AGENT DEFINITIONS 🤖
# =============================================================================

# Data Loader Agent - Handles file operations and data loading via MCP
data_loader_agent = Agent(
    name="DataLoaderAgent",
    instructions=f"""You are a Data Loader Agent specialized in file operations and data preparation.

Your expertise includes:
📁 File Operations:
- Investigating directory contents
- Loading CSV files from various sources
- Validating data format and structure
- Handling file paths and directory navigation

📊 Data Preparation:
- Initial data inspection and validation
- Column identification and data type detection
- Data quality assessment
- Preparing data for analysis workflow

🔗 MCP Integration:
- All file operations run via MCP server at {MCP_SERVER_URL}
- Use MCP tools: investigate_directory, load_csv_file, get_column_names, get_dataset_info
- Debug output shows MCP calls for transparency
- Shared data state across all agents

When loading data:
- Start by investigating available files if path is unclear
- Load CSV files and validate structure
- Provide summary of loaded data including shape and columns
- Hand off to Analytics or Visualization agents for further processing
- Alert users to any data quality issues

You are the "input" specialist - ensuring data is properly loaded and ready for analysis.""",
    model=get_model()
    # No tools specified - agents framework should auto-discover from MCP server
)

# Analytics Agent - Performs calculations and statistical analysis via MCP
analytics_agent = Agent(
    name="AnalyticsAgent", 
    instructions=f"""You are an Analytics Agent specialized in statistical analysis and data calculations.

Your expertise includes:
📊 Statistical Analysis:
- Calculating averages, medians, and comprehensive statistics
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

🔗 MCP Integration:
- All calculations run via MCP server at {MCP_SERVER_URL}
- Use MCP tools: calculate_column_average, calculate_column_stats, count_rows_with_value, find_max_value, find_min_value, group_by_column_and_aggregate, filter_data
- Debug output shows MCP calls for transparency
- Shared data state across all agents

When performing analysis:
- Ensure data is properly loaded before calculations
- Handle non-numeric data gracefully
- Provide context with your calculations (not just numbers)
- Hand off to Communication Agent for user-friendly presentation
- Suggest related analyses that might be valuable

You focus on the "processing" side - turning raw data into meaningful insights.""",
    model=get_model()
    # No tools specified - agents framework should auto-discover from MCP server
)

# Data Visualization Agent - Creates charts and visual representations via MCP
visualization_agent = Agent(
    name="VisualizationAgent",
    instructions=f"""You are a Data Visualization Agent specialized in creating meaningful visual representations of data.

Your expertise includes:
📊 Chart Creation:
- Bar charts for comparing categories and values
- Line charts for time series and trends (when implemented)
- Distribution plots and histograms (when implemented)
- Multi-series visualizations for comparisons

🎨 Visualization Design:
- Selecting appropriate chart types for data
- Proper labeling and formatting
- Color schemes and styling
- Clear titles and legends

📈 Visual Analytics:
- Identifying patterns through visualization
- Highlighting key insights visually
- Comparing multiple data series
- Showing distributions and outliers

🔗 MCP Integration:
- All charts generated via MCP server at {MCP_SERVER_URL}
- Use MCP tools: create_bar_chart, create_line_chart (when available)
- Debug output shows MCP calls for transparency
- Shared data state across all agents

When creating visualizations:
- Ensure data is properly loaded and prepared
- Choose the most appropriate visualization type
- Provide clear, descriptive titles and labels
- Save charts with meaningful filenames
- Hand off to Communication Agent for presentation
- Suggest additional visualizations that might be valuable

You focus on making data visual and accessible - turning numbers into insights through charts.""",
    model=get_model()
    # No tools specified - agents framework should auto-discover from MCP server
)

# Communication Agent - Formats responses in user-friendly language
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
    # Communication agent doesn't need direct data tools - it formats output from other agents
)

# Set up handoffs between agents
data_loader_agent.handoffs = [analytics_agent, visualization_agent, communication_agent]
analytics_agent.handoffs = [data_loader_agent, visualization_agent, communication_agent]
visualization_agent.handoffs = [data_loader_agent, analytics_agent, communication_agent]
communication_agent.handoffs = [data_loader_agent, analytics_agent, visualization_agent]

# Agent registry for multi-agent system
multi_agents = {
    "data_loader": data_loader_agent,
    "analytics": analytics_agent,
    "visualization": visualization_agent,
    "communication": communication_agent
}

# =============================================================================
# LLM-AS-A-JUDGE SYSTEM 🏛️⚖️
# =============================================================================

# Judge Agent - Evaluates response quality using different model
judge_agent = Agent(
    name="JudgeAgent",
    instructions="""You are an LLM Judge that evaluates the quality of data analysis responses.

Your role is to assess responses on multiple dimensions:

📊 **Technical Accuracy**:
- Are the statistical calculations correct?
- Are the data interpretations sound?
- Are the visualizations appropriate for the data type?

🎯 **Completeness**:
- Does the response fully address the user's question?
- Are important insights highlighted?
- Are follow-up suggestions provided?

💬 **Communication Quality**:
- Is the explanation clear and understandable?
- Is the tone appropriate and engaging?
- Are technical concepts explained accessibly?

📈 **Actionability**:
- Does the response provide actionable insights?
- Are next steps or follow-up analyses suggested?
- Is the information presented in a useful way?

**Evaluation Process**:
1. Read the original user question
2. Review the agent's response
3. Score each dimension (1-5 scale)
4. Provide specific feedback for improvement
5. Give an overall assessment

**Output Format**:
- Technical Accuracy: X/5 - [brief explanation]
- Completeness: X/5 - [brief explanation]  
- Communication: X/5 - [brief explanation]
- Actionability: X/5 - [brief explanation]
- Overall Score: X/5
- Key Strengths: [bullet points]
- Areas for Improvement: [bullet points]""",
    model=get_judge_model(),
    tools=[]
)

# =============================================================================
# CONVERSATION MANAGEMENT 💬
# =============================================================================

def print_system_info():
    """Print system information and capabilities."""
    print("\n" + "="*80)
    print("🤖 MULTI-AGENT CSV DATA ANALYSIS & VISUALIZATION SYSTEM")
    print("="*80)
    print()
    print("🎯 System Capabilities:")
    print("• 📁 Data Loading: CSV files and directory exploration")
    print("• 📊 Analytics: Statistical analysis and calculations")
    print("• 📈 Visualization: Charts and plots")
    print()
    print("🎨 Visualization Capabilities:")
    print("• 📈 Bar Charts - Compare categories")
    print("• 🔍 Scatter Plots - Explore relationships")
    print("• 📦 Box Plots - Detect outliers")
    print()
    print("💡 Example commands:")
    print('   "Load employee_data.csv and analyze the data"')
    print('   "Create a scatter plot of salary vs performance"')
    print('   "Make a bar chart showing average salary by department"')
    print('   "What are the key insights from this data?"')
    print()
    print("🔗 MCP Integration:")
    print(f"   Server URL: {MCP_SERVER_URL}")
    print("   All tools centralized via MCP protocol")
    print()

# =============================================================================
# GRADIO INTERFACE 🌐
# =============================================================================

def create_gradio_interface():
    """Create and configure the Gradio interface for the multi-agent system."""
    
    print("\n🌐 Setting up Gradio Interface...")
    
    # Initialize the SQLite session for conversation memory
    session = SQLiteSession("multi_agent_conversations.db")
    
    # Initialize the runner with the main agent
    runner = Runner(data_loader_agent, session=session)
    
    async def chat_response(message, history):
        """Process user message and return agent response."""
        try:
            print(f"\n🗣️ User: {message}")
            
            # Run the agent and get response
            response = await runner.run(message)
            
            print(f"🤖 Agent Response: {response}")
            
            return response
            
        except Exception as e:
            error_msg = f"❌ Error processing request: {str(e)}"
            print(error_msg)
            return error_msg
    
    # Create the Gradio interface
    interface = gr.ChatInterface(
        fn=chat_response,
        title="🤖 Multi-Agent CSV Data Analysis & Visualization System",
        description="""
        ### 🎯 Intelligent Data Analysis with MCP Integration
        
        Upload and analyze CSV files with natural language queries! This system uses specialized agents:
        
        - **📁 Data Loader**: Handles file operations and data loading
        - **📊 Analytics**: Performs statistical analysis and calculations  
        - **📈 Visualization**: Creates charts and visual insights
        - **💬 Communication**: Formats responses in user-friendly language
        
        **💡 Try asking:**
        - "Load employee_data.csv and show me the structure"
        - "What's the average salary by department?"
        - "Create a bar chart of sales by region"
        - "Find the top 5 performers and visualize them"
        
        🔗 **MCP Integration**: All tools are centralized via Model Context Protocol for better performance and debugging.
        """,
        examples=[
            "Load the employee data and tell me about it",
            "What's the average salary in the dataset?", 
            "Create a bar chart showing sales by department",
            "Who are the top 5 performers?",
            "Show me salary distribution by department",
            "Create a visualization comparing performance scores"
        ],
        cache_examples=False,
        theme=gr.themes.Soft(),
        additional_inputs=[]
    )
    
    # Print system information
    print_system_info()
    
    return interface

if __name__ == "__main__":
    """Run the Gradio interface when script is executed directly."""
    interface = create_gradio_interface()
    
    # Get configuration from environment variables for Docker compatibility
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    
    print(f"\n🚀 Starting Gradio server...")
    print(f"   🌐 Server: {server_name}:{server_port}")
    print(f"   📱 Access: http://localhost:{server_port}")
    
    interface.launch(
        server_name=server_name,
        server_port=server_port,
        share=False
    )
