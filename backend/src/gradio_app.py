"""
Gradio interface for the MCP-based multi-agent system
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

def get_example_questions():
    """Get example questions to display in chat initially"""
    return [
        "Investigate the data directory and show available files",
        "Load employee_data.csv and show basic statistics", 
        "Create scatter plot with salary and hire date for employees",
        "Find the maximum salary and show which employee has it",
        "Group employees by department and calculate average salary",
        "Create a bar chart showing salary distribution by department"
    ]

def create_gradio_interface():
    """Create Gradio interface for the MCP-based multi-agent system"""
    
    # Session state
    session_state = {
        "session_id": f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "judge_evaluation": False
    }
    
    async def process_query_async(user_input, history, gallery, enable_judge=False):
        """Process user query with the multi-agent system"""
        if not user_input.strip():
            return history, "", gallery
        
        # Add user message to history
        history = history or []
        history.append([user_input, "🤖 Processing with multi-agent system..."])
        
        try:
            # Get response from MCP-based multi-agent system
            agent_response = await run_multi_agent_analysis(
                user_input, 
                session_state["session_id"],
                enable_judge=enable_judge
            )
            
            # Update history with actual response
            history[-1][1] = agent_response
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history[-1][1] = error_msg
        
        # Update gallery with latest charts
        updated_gallery = get_chart_gallery()
        
        return history, "", updated_gallery
    
    def process_query(user_input, history, gallery, enable_judge=False):
        """Sync wrapper for Gradio"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_query_async(user_input, history, gallery, enable_judge))
        finally:
            loop.close()
    
    def reset_session():
        """Reset the conversation session"""
        session_state["session_id"] = f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        gallery = get_chart_gallery()
        
        # Show example questions in chat
        initial_history = []
        examples = get_example_questions()
        example_text = "Welcome to the **Data Analysis Assistant**! 🤖\n\n"
        example_text += "Ask natural questions about your CSVs and get concise, actionable replies.\n\n"
        example_text += "**Example questions you can ask:**\n"
        for i, example in enumerate(examples, 1):
            example_text += f"{i}. {example}\n"
        
        initial_history.append([None, example_text])
        return initial_history, "", gallery, False  # Also reset judge checkbox
    
    # Create custom dark theme similar to the interface shown
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
    
    # Create the interface
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
        .main-header h1 {
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .main-header p {
            font-size: 1rem;
            opacity: 0.8;
        }
        .chart-gallery {
            background: #1e2329;
            border-radius: 8px;
            padding: 15px;
        }
        .chat-container {
            background: #1e2329;
            border-radius: 8px;
            padding: 15px;
        }
        """
    ) as interface:
        
        # Initialize gallery with existing charts
        initial_gallery = get_chart_gallery()
        
        # Initialize with example questions
        initial_history, _, _, _ = reset_session()
        
        gr.HTML("""
        <div class="main-header">
            <h1>Data Analysis Assistant</h1>
            <p>Ask natural questions about your CSVs and get concise, actionable replies.</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=3, elem_classes=["chat-container"]):
                chatbot = gr.Chatbot(
                    value=initial_history,
                    label="💬 Chatbot",
                    height=500,
                    show_label=False,
                    container=True,
                    bubble_full_width=False,
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask a question (e.g., 'Load employee data.csv and show key stats')",
                        label="",
                        lines=1,
                        scale=4,
                        show_label=False,
                        container=False
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1, size="sm")
                
                with gr.Row():
                    judge_checkbox = gr.Checkbox(
                        label="🏛️ Enable Judge Evaluation",
                        value=False,
                        info="Get quality assessment from Gemini judge"
                    )
                    clear_btn = gr.Button("New Session", variant="secondary", size="sm")
                
            with gr.Column(scale=1, elem_classes=["chart-gallery"]):
                gr.HTML("<h3 style='color: white; margin-bottom: 15px; text-align: center;'>📊 Visualizations</h3>")
                
                with gr.Row():
                    gr.HTML("<p style='color: #d1d5db; text-align: center; margin-bottom: 10px;'>Gallery</p>")
                
                gallery = gr.Gallery(
                    value=initial_gallery,
                    label="",
                    show_label=False,
                    columns=2,
                    rows=3,
                    height=400,
                    container=True,
                    preview=True,
                    object_fit="cover"
                )
                
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[msg, chatbot, gallery, judge_checkbox],
            outputs=[chatbot, msg, gallery]
        )
        
        msg.submit(
            fn=process_query,
            inputs=[msg, chatbot, gallery, judge_checkbox],
            outputs=[chatbot, msg, gallery]
        )
        
        clear_btn.click(
            fn=reset_session,
            outputs=[chatbot, msg, gallery, judge_checkbox]
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
