"""
Gradio interface for the MCP-based agent system
"""

import gradio as gr
import asyncio
import sys
import os
import glob
from pathlib import Path

# Add the src directory to the Python path for proper imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.multi_agent_mcp import run_multi_agent_analysis
from datetime import datetime
from phoenix.otel import register

from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().uninstrument()

# Enable Phoenix tracing for monitoring agent interactions
phoenix_endpoint = os.getenv("PHOENIX_ENDPOINT", "http://phoenix:6006/v1/traces")
phoenix_project_name = os.getenv("PHOENIX_PROJECT_NAME", "mcp-gradio-interface")

# Session state for maintaining conversation context
session_state = {
    "session_id": f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "messages": [],
    "judge_evaluation": False
}

def get_example_questions():
    """Return a list of example questions for the chat interface."""
    return [
        "Load employee_data.csv and show basic statistics",
        "Create scatter plot with salary and hire date for employees", 
        "Find the maximum salary and show which employee has it",
        "Create a bar chart showing salary distribution by department"
    ]
phoenix_project_name = os.getenv("PHOENIX_PROJECT_NAME", "agentic-app-quickstart")

# Charts directory path
CHARTS_DIR = Path("/app/charts")


def get_chart_gallery():
    """Get list of chart images from charts directory"""
    if not CHARTS_DIR.exists():
        CHARTS_DIR.mkdir(exist_ok=True)
        return []
    
    chart_files = list(CHARTS_DIR.glob("*.png"))
    chart_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)  # Most recent first
    return [str(chart) for chart in chart_files[:12]]  # Show up to 12 most recent charts

def remove_duplicate_function():
    """Remove duplicate function definition"""
    pass

def create_gradio_interface():
    """Create Gradio interface using ChatInterface for the MCP-based multi-agent system"""
    
    # Session state
    session_state = {
        "session_id": f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "judge_evaluation": False
    }
    
    async def process_query_async(user_input, enable_judge=False):
        """Process user query with the multi-agent system"""
        if not user_input.strip():
            return "Please enter a question."
        
        try:
            # Get response from MCP-based multi-agent system
            agent_response = await run_multi_agent_analysis(
                user_input, 
                session_state["session_id"],
                enable_judge=enable_judge
            )
            
            return agent_response
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def chat(message, history):
        """Chat function for ChatInterface"""
        if not message.strip():
            return "Please enter a question."
        
        # Get judge evaluation setting from session state
        enable_judge = session_state.get("judge_evaluation", False)
        
        # Process the query
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(process_query_async(message, enable_judge))
            # Update gallery after processing
            updated_gallery = get_chart_gallery()
            return response, gr.update(value=updated_gallery)
        finally:
            loop.close()
    
    # Create custom dark theme
    custom_theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f1419",
        background_fill_primary="#1e2329",
        background_fill_secondary="#2a2f3a",
        block_background_fill="#1e2329",
        block_border_color="#3a4454",
        button_primary_background_fill="#6366f1",
        button_primary_text_color="white",
        button_secondary_background_fill="#374151",
        button_secondary_text_color="white",
        body_text_color="#f3f4f6",
    )
    
    # Create the interface using ChatInterface
    with gr.Blocks(
        title="Data Analysis Assistant", 
        theme=custom_theme,
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            color: white;
            margin-bottom: 20px;
        }
        .header-controls {
            background: #1e2329;
            border: 1px solid #3a4454;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
        }
        """
    ) as interface:
        
        # Initialize gallery with existing charts
        gallery = gr.State(get_chart_gallery())
        
        gr.HTML("""
        <div class="main-header">
            <h1>Data Analysis Assistant</h1>
            <p>Ask natural questions about your CSVs and get concise, actionable replies.</p>
        </div>
        """)
        
        # Header controls for LLM as Judge
        with gr.Row(elem_classes=["header-controls"]):
            judge_checkbox = gr.Checkbox(
                label="🏛️ Enable Judge Evaluation",
                value=False,
                info="Get quality assessment from Gemini judge"
            )
        
        with gr.Row():
            with gr.Column(scale=3):
                # Use ChatInterface with examples
                chat_interface = gr.ChatInterface(
                    chat,
                    examples=get_example_questions(),
                    additional_outputs=[gallery],
                    type="messages",
                    title="💬 Chat with Data Analysis Assistant"
                )
                
                # Add judge checkbox change handler
                def update_judge_setting(enabled):
                    session_state["judge_evaluation"] = enabled
                    return f"Judge evaluation {'enabled' if enabled else 'disabled'}"
                
                judge_checkbox.change(
                    fn=update_judge_setting,
                    inputs=[judge_checkbox],
                    outputs=[]
                )
            
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: white; margin-bottom: 15px; text-align: center;'>📊 Visualizations</h3>")
                
                gallery_display = gr.Gallery(
                    value=get_chart_gallery(),
                    label="",
                    show_label=False,
                    columns=2,
                    rows=3,
                    height=500,
                    container=True,
                    preview=True,
                    object_fit="cover"
                )
                
                # Connect gallery updates
                gallery.change(
                    fn=lambda x: x,
                    inputs=[gallery],
                    outputs=[gallery_display]
                )
    
    return interface


if __name__ == "__main__":

    print(f"🔍 Initializing Phoenix tracing...")
    print(f"   📡 Endpoint: {phoenix_endpoint}")
    print(f"   📁 Project: {phoenix_project_name}")

    tracer_provider = register(
        endpoint=phoenix_endpoint,
        project_name=phoenix_project_name,
        auto_instrument=True
    )


    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
