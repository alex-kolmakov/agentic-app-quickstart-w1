"""
Multi-Agent CSV Data Analysis System with Visualization

This module implements a specialized multi-agent approach for CSV data analysis with
advanced visualization capabilities. It provides intelligent systems that can load CSV files,
analyze data, create insightful visualizations, and answer natural language qu    print("• 📈 Visualization: Creates charts and plots")
    print()
    print("🎨 Visualization Capabilities:")
    print("• 📈 Bar Charts - Compare categories")
    print("• 🔍 Scatter Plots - Explore relationships")
    print("• 📦 Box Plots - Detect outliers")
    print()
    print("💡 Example commands:")
    print('   "Load employee_data.csv and analyze the data"')
    print('   "Create a scatter plot of salary vs performance"')
    print('   "Make a bar chart showing average salary by department"')eatures:
- 🤖🤖🤖 Multi-Agent System: Specialized agents for different aspects
- 📈 Data Visualization: Charts and plots
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
from phoenix.otel import register
from multi_agent_data_app.helpers.helpers import get_model
from multi_agent_data_app.helpers.google_helpers import get_model as get_judge_model
from datetime import datetime
import gradio as gr
import os
from multi_agent_data_app.tools import AVAILABLE_TOOLS
from dotenv import load_dotenv
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().uninstrument()

load_dotenv()

# # Enable Phoenix tracing for monitoring agent interactions
# Use environment variables for Phoenix configuration
phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
phoenix_project_name = os.getenv("PHOENIX_PROJECT_NAME", "agentic-app-quickstart")

print(f"🔍 Initializing Phoenix tracing...")
print(f"   📡 Endpoint: {phoenix_endpoint}")
print(f"   📁 Project: {phoenix_project_name}")

try:
    tracer_provider = register(
        project_name=phoenix_project_name,
        endpoint=f"{phoenix_endpoint}/v1/traces",
        batch=True,
        auto_instrument=True
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    print("   ✅ Phoenix tracing initialized successfully")
except Exception as e:
    print(f"   ⚠️ Phoenix tracing initialization failed: {e}")
    print("   📝 Continuing without tracing...")
    # Continue without tracing if Phoenix is not available

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
- Bar charts for categorical comparisons
- Scatter plots for relationship exploration
- Box plots for outlier detection and distribution

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
        'create_bar_chart', 'create_scatter_plot', 'create_box_plot'
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

💬 **Communication Quality**:
- Is the response clear and understandable?
- Are technical concepts explained appropriately?
- Is the tone helpful and engaging?

🎯 **Completeness**:
- Does the response fully address the user's question?
- Are important insights highlighted?
- Are relevant follow-up suggestions provided?

🔍 **Professional Standards**:
- Is the analysis methodologically sound?
- Are limitations or assumptions acknowledged?
- Is the confidence level appropriate?

Provide your evaluation as a score from 1-10 and brief constructive feedback.
Focus on being helpful and educational rather than purely critical.

Return your judgment in this format:
**Quality Score: X/10**
**Strengths:** [Key strengths]
**Areas for Improvement:** [Specific suggestions]
**Overall Assessment:** [Brief summary]""",
    model=get_judge_model(),  # Using Google Gemini as judge
    tools=[]
)

async def judge_response(user_question: str, agent_response: str, session_id: str = "judge_session") -> str:
    """
    Evaluate the quality of an agent response using LLM-as-a-judge.
    
    Args:
        user_question (str): Original user question
        agent_response (str): Response from the multi-agent system
        session_id (str): Session ID for judge evaluation
        
    Returns:
        str: Judge's evaluation and feedback
    """
    try:
        # Create judge session
        judge_session = SQLiteSession(session_id=f"judge_{session_id}")
        
        # Construct evaluation prompt
        evaluation_prompt = f"""Please evaluate this data analysis response:

**User Question:** {user_question}

**Agent Response:** {agent_response}

Please provide your assessment focusing on technical accuracy, communication quality, completeness, and professional standards."""

        # Get judge evaluation
        judge_result = await Runner.run(
            starting_agent=judge_agent,
            input=evaluation_prompt,
            session=judge_session
        )
        
        return judge_result.final_output
        
    except Exception as e:
        return f"⚖️ Judge evaluation unavailable: {str(e)}"


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
            
            # Display agent response  
            print(f"\n🤖 Agent: {result.final_output}")
            
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or rephrase your question.")

# =============================================================================
# ENHANCED CONVERSATION FUNCTION WITH TRACKING
# =============================================================================

async def run_enhanced_conversation(user_input, session_id="gradio_session", include_judge=True):
    """
    Run enhanced conversation with optional judge evaluation
    
    Args:
        user_input (str): User's question or command
        session_id (str): Session identifier for memory
        include_judge (bool): Whether to include judge evaluation
        
    Returns:
        tuple: (agent_response, files_created, judge_evaluation)
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
        
        # Check for generated files (look for common image extensions)
        files_created = []
        # Use the charts directory that's mounted as a volume in Docker
        charts_dir = "/app/charts"
        
        try:
            if os.path.exists(charts_dir):
                for file in os.listdir(charts_dir):
                    if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                        files_created.append(os.path.join(charts_dir, file))
                
                # Sort by creation time (newest first)
                if files_created:
                    files_created.sort(key=os.path.getctime, reverse=True)
        except Exception as e:
            print(f"Warning: Could not check for generated charts: {e}")
            files_created = []
        
        # Get judge evaluation if requested
        judge_evaluation = ""
        if include_judge and result.final_output:
            judge_evaluation = await judge_response(user_input, result.final_output, session_id)
        
        return result.final_output, files_created, judge_evaluation
        
    except Exception as e:
        error_msg = f"❌ An error occurred: {str(e)}"
        return error_msg, [], ""

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_gradio_interface():
    """
    Create and configure the Gradio web interface
    """
    
    # State management for session
    session_state = {"session_id": f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    
    async def process_query(user_input, history, enable_judge):
        """Process user query and return response with optional judge evaluation"""
        if not user_input.strip():
            return history, ""
        
        # Add user message to history
        history = history or []
        history.append([user_input, "Processing..."])
        
        # Get response from agents with optional judge evaluation
        agent_response, files_created, judge_evaluation = await run_enhanced_conversation(
            user_input, 
            session_state["session_id"],
            include_judge=enable_judge
        )
        
        # Format complete response
        complete_response = agent_response
        
        # Add judge evaluation if enabled and available
        if enable_judge and judge_evaluation:
            complete_response += f"\n\n⚖️ **JUDGE EVALUATION**\n{judge_evaluation}"
        
        # Add files information if any were created
        if files_created:
            chart_info = "\n\n📊 **Visualizations Generated:**\n"
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
        
        **New Feature**: ⚖️ **LLM-as-a-Judge** evaluation using Google Gemini for response quality assessment!
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
                
                # Judge evaluation toggle
                judge_toggle = gr.Checkbox(
                    label="⚖️ Enable LLM Judge Evaluation (Gemini)",
                    value=True,
                    info="Get quality assessment of responses from an independent judge model"
                )
            
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
                **Example requests:**
                - *"Create a bar chart of department vs salary"*
                - *"Show me a scatter plot of performance vs experience"*
                - *"Analyze the data and show basic statistics"*
                - *"Make a box plot to detect outliers"*
                """)
                
        
        # Event handlers
        def handle_submit(user_input, history, enable_judge):
            """Handle form submission synchronously"""
            try:
                # Run the async process_query function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    history, _ = loop.run_until_complete(process_query(user_input, history, enable_judge))
                finally:
                    loop.close()
                
                # Get any generated visualizations from the charts directory
                charts_dir = "/app/charts"
                if os.path.exists(charts_dir):
                    image_files = []
                    for file in os.listdir(charts_dir):
                        if file.endswith(('.png', '.jpg', '.jpeg', '.svg')):
                            image_files.append(os.path.join(charts_dir, file))
                    
                    # Sort by creation time (newest first)  
                    if image_files:
                        image_files.sort(key=os.path.getctime, reverse=True)
                        return history, "", image_files
                
                return history, "", None
                
            except Exception as e:
                print(f"Error in handle_submit: {e}")
                history = history or []
                history.append([user_input, f"❌ Error processing request: {str(e)}"])
                return history, "", None
        
        submit_btn.click(
            handle_submit,
            inputs=[msg, chatbot, judge_toggle],
            outputs=[chatbot, msg, image_gallery]
        )
        
        msg.submit(
            handle_submit,
            inputs=[msg, chatbot, judge_toggle], 
            outputs=[chatbot, msg, image_gallery]
        )
        
        clear_btn.click(
            reset_session,
            outputs=[chatbot, msg, image_gallery]
        )
    
    return interface