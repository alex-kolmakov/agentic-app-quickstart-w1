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
from datetime import datetime
import gradio as gr
import os
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
- Investigating directories to find appropriate data files
- Loading CSV files from various paths
- Validating file formats and structure
- Handling file errors and providing clear feedback
- Inspecting dataset metadata (shape, columns, types)

🔍 Data Inspection:
- Providing dataset summaries and overviews
- Listing column names and data types
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
    model=get_model(),
    tools=[tool for tool in AVAILABLE_TOOLS if tool.name in [
        'investigate_directory', 'load_csv_file', 'get_column_names', 'get_dataset_info', 'get_unique_values'
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
    print("• 📈 Bar Charts - Compare categories")
    print("• 🔍 Scatter Plots - Explore relationships")
    print("• 📦 Box Plots - Detect outliers")
    print()
    print("💡 Example commands:")
    print('   "Load employee_data.csv and create a dashboard"')
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
            breakpoint()
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



# =============================================================================
# HANDOFF TRACKING UTILITIES
# =============================================================================

def extract_handoff_map(result):
    """
    Extract and analyze agent handoffs from result.new_items
    
    Args:
        result: RunResult object from agents framework
        
    Returns:
        dict: Handoff analysis with timeline and agent interactions
    """
    handoff_map = {
        "agents_involved": [],
        "handoff_sequence": [],
        "agent_contributions": {},
        "timeline": [],
        "total_handoffs": 0
    }
    
    if not hasattr(result, 'new_items') or not result.new_items:
        return handoff_map
    
    for i, item in enumerate(result.new_items):
        if hasattr(item, 'agent') and item.agent:
            agent_name = item.agent.name
            
            # Track unique agents
            if agent_name not in handoff_map["agents_involved"]:
                handoff_map["agents_involved"].append(agent_name)
            
            # Track sequence of agent activations
            handoff_map["handoff_sequence"].append(agent_name)
            
            # Count contributions per agent
            if agent_name not in handoff_map["agent_contributions"]:
                handoff_map["agent_contributions"][agent_name] = 0
            handoff_map["agent_contributions"][agent_name] += 1
            
            # Create timeline entry
            timeline_entry = {
                "step": i + 1,
                "agent": agent_name,
                "timestamp": datetime.now().isoformat(),
                "item_type": type(item).__name__
            }
            handoff_map["timeline"].append(timeline_entry)
    
    # Calculate handoffs (transitions between different agents)
    handoffs = 0
    for i in range(1, len(handoff_map["handoff_sequence"])):
        if handoff_map["handoff_sequence"][i] != handoff_map["handoff_sequence"][i-1]:
            handoffs += 1
    
    handoff_map["total_handoffs"] = handoffs
    
    return handoff_map

def format_handoff_summary(handoff_map):
    """
    Format the handoff map into a human-readable summary
    
    Args:
        handoff_map (dict): Result from extract_handoff_map
        
    Returns:
        str: Formatted summary of agent interactions
    """
    if not handoff_map["agents_involved"]:
        return "🤖 No agent interactions detected."
    
    summary = []
    summary.append("🔄 **AGENT HANDOFF ANALYSIS**")
    summary.append("=" * 40)
    
    # Agent involvement
    summary.append(f"👥 **Agents Involved:** {len(handoff_map['agents_involved'])}")
    for agent in handoff_map["agents_involved"]:
        emoji = {
            "DataLoaderAgent": "📁",
            "AnalyticsAgent": "📊", 
            "VisualizationAgent": "📈",
            "CommunicationAgent": "💬"
        }.get(agent, "🤖")
        contributions = handoff_map["agent_contributions"].get(agent, 0)
        summary.append(f"  {emoji} {agent}: {contributions} contribution(s)")
    
    # Handoff sequence
    summary.append(f"\n🔄 **Total Handoffs:** {handoff_map['total_handoffs']}")
    
    if len(handoff_map["handoff_sequence"]) > 1:
        summary.append("📋 **Agent Sequence:**")
        sequence_str = " → ".join([
            {
                "DataLoaderAgent": "📁 DataLoader",
                "AnalyticsAgent": "📊 Analytics", 
                "VisualizationAgent": "📈 Visualization",
                "CommunicationAgent": "💬 Communication"
            }.get(agent, f"🤖 {agent}") 
            for agent in handoff_map["handoff_sequence"]
        ])
        summary.append(f"  {sequence_str}")
    
    # Timeline
    if handoff_map["timeline"]:
        summary.append("\n⏱️ **Execution Timeline:**")
        for entry in handoff_map["timeline"]:
            emoji = {
                "DataLoaderAgent": "📁",
                "AnalyticsAgent": "📊", 
                "VisualizationAgent": "📈",
                "CommunicationAgent": "💬"
            }.get(entry["agent"], "🤖")
            summary.append(f"  Step {entry['step']}: {emoji} {entry['agent']}")
    
    return "\n".join(summary)

# =============================================================================
# ENHANCED CONVERSATION FUNCTION WITH TRACKING
# =============================================================================

async def run_enhanced_conversation(user_input, session_id="gradio_session"):
    """
    Run enhanced conversation with handoff tracking
    
    Args:
        user_input (str): User's question or command
        session_id (str): Session identifier for memory
        
    Returns:
        tuple: (agent_response, handoff_summary, files_created)
    """
    try:
        # Create session for memory
        session = SQLiteSession(session_id=session_id)
        
        # Start with data loader agent as the entry point
        result = await Runner.run(
            starting_agent=data_loader_agent,
            input=user_input,
            session=session
        )
        
        # Extract handoff information
        handoff_map = extract_handoff_map(result)
        handoff_summary = format_handoff_summary(handoff_map)
        
        # Check for generated files (look for common image extensions)
        import os
        files_created = []
        solution_dir = "/Users/helloworld/Projects/agentic-app-quickstart-w1/agentic_app_quickstart/week_1/solution"
        for file in os.listdir(solution_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.svg')) and file not in ['employee_dashboard.png']:
                files_created.append(os.path.join(solution_dir, file))
        
        # Sort by creation time (newest first)
        files_created.sort(key=os.path.getctime, reverse=True)
        
        return result.final_output, handoff_summary, files_created
        
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        return error_msg, "🔄 No handoff analysis available due to error.", []

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_gradio_interface():
    """
    Create and configure the Gradio web interface
    """
    
    # State management for session
    session_state = {"session_id": f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    
    async def process_query(user_input, history):
        """Process user query and return response with handoff analysis"""
        if not user_input.strip():
            return history, ""
        
        # Add user message to history
        history = history or []
        history.append([user_input, "Processing..."])
        
        # Get response from agents
        agent_response, handoff_summary, files_created = await run_enhanced_conversation(
            user_input, 
            session_state["session_id"]
        )
        
        # Format complete response
        complete_response = f"{agent_response}\n\n{handoff_summary}"
        
        # Add files information if any were created
        if files_created:
            chart_info = "\n\n� **Visualizations Generated:**\n"
            for file_path in files_created:
                file_name = os.path.basename(file_path)
                chart_info += f"  🖼️ {file_name}\n"
            complete_response += chart_info + "\n*(Visualizations are displayed in the gallery panel)*"
        
        # Update history with actual response
        history[-1][1] = complete_response
        
        return history, ""
    
    def reset_session():
        """Reset the conversation session"""
        session_state["session_id"] = f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return [], "🔄 Session reset! New conversation started.", None
    
    # Create the interface
    with gr.Blocks(title="🎪📈 Multi-Agent CSV Analysis System") as interface:
        
        gr.Markdown("""
        # 🎪📈 Multi-Agent CSV Data Analysis & Visualization System
        
        Welcome to your intelligent data analysis assistant! This system uses **4 specialized agents** working together:
        - 📁 **DataLoader**: File operations and data preparation  
        - 📊 **Analytics**: Statistical calculations and analysis
        - 📈 **Visualization**: Chart creation and visual insights
        - 💬 **Communication**: User-friendly response formatting
        """)
        
        with gr.Row():
            # Chat column
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    label="💬 Conversation with Multi-Agent System",
                    show_label=True
                )
                
                msg = gr.Textbox(
                    placeholder="Ask about your data... (e.g., 'Load employee_data.csv and analyze it')",
                    label="Your Question",
                    lines=2
                )
                
                with gr.Row():
                    submit_btn = gr.Button("🚀 Analyze", variant="primary")
                    clear_btn = gr.Button("🔄 Reset Session", variant="secondary")
            
            # Visualization column with gallery
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Data Visualizations")
                image_gallery = gr.Gallery(
                    label="Generated Charts & Plots",
                    show_label=True,
                    columns=1,
                    height=500,
                    object_fit="contain"
                )
                gr.Markdown("""
                *Ask the system to create visualizations like:*
                - "Create a bar chart of department vs salary"
                - "Show me a scatter plot of performance vs experience"
                - "Make a histogram of employee ages"
                """)
                
        
        # Event handlers
        def handle_submit(user_input, history):
            history, _ = asyncio.run(process_query(user_input, history))
            
            # Get any generated visualizations
            solution_dir = "/Users/helloworld/Projects/agentic-app-quickstart-w1/agentic_app_quickstart/week_1/solution"
            image_files = []
            for file in os.listdir(solution_dir):
                if file.endswith(('.png', '.jpg', '.jpeg', '.svg')) and file not in ['employee_dashboard.png']:
                    image_files.append(os.path.join(solution_dir, file))
            
            # Sort by creation time (newest first)
            image_files.sort(key=os.path.getctime, reverse=True)
            
            return history, "", image_files if image_files else None
        
        submit_btn.click(
            handle_submit,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, image_gallery]
        )
        
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot], 
            outputs=[chatbot, msg, image_gallery]
        )
        
        clear_btn.click(
            reset_session,
            outputs=[chatbot, msg, image_gallery]
        )
    
    return interface


if __name__ == "__main__":
    asyncio.run(run_demo())
