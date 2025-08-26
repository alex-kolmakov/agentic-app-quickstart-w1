"""
Gradio interface for the MCP-based multi-agent system
"""

import gradio as gr
import asyncio
from .agent.multi_agent_mcp import run_multi_agent_analysis
from datetime import datetime
import os


def create_gradio_interface():
    """Create Gradio interface for the MCP-based multi-agent system"""
    
    # Session state
    session_state = {"session_id": f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"}
    
    async def process_query_async(user_input, history):
        """Process user query with the multi-agent system"""
        if not user_input.strip():
            return history, ""
        
        # Add user message to history
        history = history or []
        history.append([user_input, "🤖 Processing with multi-agent system..."])
        
        try:
            # Get response from MCP-based multi-agent system
            agent_response = await run_multi_agent_analysis(
                user_input, 
                session_state["session_id"]
            )
            
            # Update history with actual response
            history[-1][1] = agent_response
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            history[-1][1] = error_msg
        
        return history, ""
    
    def process_query(user_input, history):
        """Sync wrapper for Gradio"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(process_query_async(user_input, history))
        finally:
            loop.close()
    
    def reset_session():
        """Reset the conversation session"""
        session_state["session_id"] = f"gradio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        return [], "🔄 Session reset! New conversation started."
    
    # Create the interface
    with gr.Blocks(title="🤖 MCP Multi-Agent Data Analysis", theme=gr.themes.Soft()) as interface:
        
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>🤖 MCP Multi-Agent Data Analysis System</h1>
            <p>Powered by <strong>Model Context Protocol (MCP)</strong> for secure data operations</p>
            <p><em>4 Specialized Agents: DataLoader • Analytics • Visualization • Communication</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="💬 Multi-Agent Conversation",
                    height=500,
                    placeholder="Ask me about your data...",
                    show_label=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="e.g., 'Load employee_data.csv and show statistics'",
                        label="Your Question",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("🚀 Analyze", variant="primary", scale=1)
                
                clear_btn = gr.Button("🔄 Reset Session", variant="secondary")
                
                # Example queries
                gr.Examples(
                    examples=[
                        "Investigate the data directory and show available files",
                        "Load employee_data.csv and show basic statistics",
                        "Find the maximum salary and show which employee has it",
                        "Group employees by department and calculate average salary",
                        "Create a bar chart showing salary distribution",
                        "Filter data for employees with salary greater than 50000"
                    ],
                    inputs=msg,
                    label="💡 Example Questions"
                )
            
            with gr.Column(scale=1):
                gr.HTML("""
                <div style="padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
                    <h3>🏗️ MCP Architecture</h3>
                    <p><strong>Secure separation of concerns:</strong></p>
                    <ul>
                        <li>🤖 <strong>Agents</strong>: AI reasoning & orchestration</li>
                        <li>🔒 <strong>MCP Server</strong>: Data operations & tools</li>
                        <li>🌐 <strong>Communication</strong>: Native MCP protocol</li>
                    </ul>
                    
                    <h3>👥 Agent Specialists</h3>
                    <ul>
                        <li>📁 <strong>DataLoader</strong>: File operations</li>
                        <li>📊 <strong>Analytics</strong>: Statistical analysis</li>
                        <li>📈 <strong>Visualization</strong>: Chart creation</li>
                        <li>💬 <strong>Communication</strong>: User-friendly responses</li>
                    </ul>
                    
                    <h3>🛠️ Available Tools</h3>
                    <ul>
                        <li>📂 investigate_directory</li>
                        <li>📄 load_csv_file</li>
                        <li>📊 calculate_column_stats</li>
                        <li>📈 create_bar_chart</li>
                        <li>🔍 filter_data</li>
                        <li>📦 group_by_aggregate</li>
                        <li>...and 9 more!</li>
                    </ul>
                </div>
                """)
        
        # Event handlers
        submit_btn.click(
            fn=process_query,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            fn=process_query,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            fn=reset_session,
            outputs=[chatbot, msg]
        )
    
    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False
    )
