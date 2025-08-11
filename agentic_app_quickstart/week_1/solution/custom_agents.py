"""
Multi-Agent CSV Data Analysis System with Visualization

This module implements a specialized multi-agent approach for CSV data analysis with
advanced visualization capabilities. It provides intelligent systems that can load CSV files,
analyze data, create insightful visualizations, and answer natural language questions.

Features:
- 🤖🤖🤖 Multi-Agent System: Specialized agents for different aspects
- 📈 Data Visualization: Charts, plots, and interactive dashboards
- 💾 Memory: Conversation history and context preservation
- 🛡️ Error Handling: Robust error handling for user-friendly experience
- 🗣️ Natural Language Interface: Understands and responds to human questions

Agent Specialists:
1. 📁 DataLoaderAgent - File operations and data preparation
2. 📊 AnalyticsAgent - Statistical calculations and analysis
3. 📈 VisualizationAgent - Chart creation and visual insights
4. 💬 CommunicationAgent - User-friendly response formatting
"""

import asyncio
from agents import Agent, Runner, SQLiteSession, set_tracing_disabled
from agentic_app_quickstart.examples.google_helpers import get_model
from tools import AVAILABLE_TOOLS

# Disable detailed logging for cleaner output
set_tracing_disabled(True)


# =============================================================================
# MULTI-AGENT SYSTEM WITH VISUALIZATION 🤖🤖🤖📈
# =============================================================================

# Data Loader Agent - Handles file operations and data preparation
data_loader_agent = Agent(
    name="DataLoaderAgent",
    instructions="""You are a Data Loader Agent specialized in CSV file operations and data preparation.

Your responsibilities:
📁 File Operations:
- Loading CSV files from various paths
- Validating file formats and structure
- Handling file errors and providing clear feedback
- Inspecting dataset metadata (shape, columns, types)

🔍 Data Inspection:
- Providing dataset summaries and overviews
- Listing column names and data types
- Showing sample data for user understanding
- Identifying potential data quality issues

When working with users:
- Always verify the file path and existence before loading
- Provide clear success/error messages
- Give helpful context about the loaded dataset
- Hand off to Analytics Agent for calculations and analysis
- Hand off to Visualization Agent for chart creation
- Hand off to Communication Agent for final user presentation

You focus on the "input" side - getting data ready for analysis.
Once data is loaded and validated, transfer to appropriate specialists.""",
    model=get_model(),
    tools=[tool for tool in AVAILABLE_TOOLS if tool.name in [
        'load_csv_file', 'get_column_names', 'get_dataset_info', 'get_unique_values'
    ]]
)

# Analytics Agent - Performs calculations and statistical analysis  
analytics_agent = Agent(
    name="AnalyticsAgent", 
    instructions="""You are an Analytics Agent specialized in statistical analysis and data calculations.

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

When performing analysis:
- Ensure data is properly loaded before calculations
- Handle non-numeric data gracefully
- Provide context with your calculations (not just numbers)
- Hand off to Visualization Agent for charts and graphs
- Hand off to Communication Agent for user-friendly presentation
- Suggest related analyses that might be valuable

You focus on the "processing" side - turning raw data into meaningful insights.""",
    model=get_model(),
    tools=[tool for tool in AVAILABLE_TOOLS if tool.name in [
        'calculate_column_average', 'calculate_column_stats', 'count_rows_with_value',
        'find_max_value', 'find_min_value', 'group_by_column_and_aggregate', 'filter_data'
    ]]
)

# Visualization Agent - Creates charts, plots and visual insights
visualization_agent = Agent(
    name="VisualizationAgent",
    instructions="""You are a Visualization Agent specialized in creating insightful charts and visual representations of data.

Your expertise includes:
📈 Chart Creation:
- Histograms for distribution analysis
- Bar charts for categorical comparisons
- Scatter plots for relationship exploration
- Box plots for outlier detection and distribution
- Correlation heatmaps for pattern discovery
- Comprehensive dashboards for overview

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
    model=get_model(),
    tools=[tool for tool in AVAILABLE_TOOLS if tool.name in [
        'create_histogram', 'create_bar_chart', 'create_scatter_plot', 
        'create_box_plot', 'create_correlation_heatmap', 'create_data_summary_dashboard'
    ]]
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
    model=get_model(),
    tools=[]  # Communication agent doesn't need direct data tools
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
# CONVERSATION FUNCTION
# =============================================================================

async def run_multi_agent_conversation():
    """
    Run a conversation with the multi-agent visualization system.
    Specialized agents work together to provide comprehensive data analysis with visuals.
    """
    print("🤖🤖🤖📈 Multi-Agent Data Analysis & Visualization System")
    print("=" * 65)
    print("Welcome to our specialized team of data analysis agents!")
    print("• 📁 Data Loader: Handles file operations and data preparation")
    print("• 📊 Analytics: Performs calculations and statistical analysis") 
    print("• 📈 Visualization: Creates charts, plots, and dashboards")
    print("• 💬 Communication: Makes results user-friendly")
    print()
    print("🎨 Visualization Capabilities:")
    print("• 📊 Histograms - Show data distributions")
    print("• 📈 Bar Charts - Compare categories")
    print("• 🔍 Scatter Plots - Explore relationships")
    print("• 📦 Box Plots - Detect outliers")
    print("• 🌡️ Heatmaps - Visualize correlations")
    print("• 📋 Dashboards - Comprehensive overviews")
    print()
    print("💡 Example commands:")
    print('   "Load employee_data.csv and create a dashboard"')
    print('   "Show me a histogram of salaries"')
    print('   "Create a scatter plot of salary vs performance"')
    print('   "Make a bar chart showing average salary by department"')
    print()
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 65)
    
    # Create session for memory
    session = SQLiteSession(session_id="multi_agent_viz_session")
    
    # Start with data loader agent as the entry point
    current_agent = data_loader_agent
    
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
            print("\n👋 Goodbye! Thanks for using our multi-agent visualization system!")
            print("🎨 Check your directory for any saved charts and visualizations!")
            break
            
        if not user_input:
            continue
            
        try:
            result = await Runner.run(
                starting_agent=current_agent,
                input=user_input,
                session=session
            )
            
            # Display agent response  
            print(f"\n🤖 Agent: {result.final_output}")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or rephrase your question.")


# =============================================================================
# DEMO FUNCTION
# =============================================================================

async def run_demo():
    """
    Demonstrate the multi-agent visualization system with sample data.
    """
    print("🎪📈 Multi-Agent Data Visualization Demo")
    print("=" * 45)
    print("Choose your experience:")
    print("1. 🤖🤖🤖📈 Interactive Multi-Agent System")
    print("2. 🚀 Quick Visualization Demo")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        await run_multi_agent_conversation()
    elif choice == "2":
        # Quick demo with visualizations
        print("\n🚀 Quick Visualization Demo")
        print("Creating sample visualizations with employee data...")
        
        session = SQLiteSession(session_id="demo_viz_session")
        
        # Demo questions with visualizations
        demo_questions = [
            "Load the employee_data.csv file",
            "Create a comprehensive dashboard",
            "Show me a histogram of salaries",
            "Create a bar chart showing average salary by department",
            "Make a scatter plot of salary vs performance score",
            "Show a box plot of performance scores by department"
        ]
        
        for question in demo_questions:
            print(f"\n💬 Demo Question: {question}")
            result = await Runner.run(
                starting_agent=data_loader_agent,
                input=question,
                session=session
            )
            
            print(f"🤖 Agent: {result.final_output}")
            print("-" * 50)
        
        print("\n✨ Demo complete! Check your directory for generated visualizations.")
    else:
        print("Invalid choice. Please run the demo again.")


if __name__ == "__main__":
    asyncio.run(run_demo())
