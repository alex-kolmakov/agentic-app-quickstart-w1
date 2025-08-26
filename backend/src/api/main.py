"""
FastAPI application for the Agentic Data Analysis Backend
Now using proper multi-agent system with MCP tools
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from ..agent.multi_agent_mcp import run_multi_agent_analysis


app = FastAPI(
    title="Agentic Data Analysis Backend",
    description="Multi-agent data analysis system using MCP for secure data operations",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class AnalysisResponse(BaseModel):
    result: str
    session_id: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agentic Multi-Agent Data Analysis Backend is running"}


@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Test connection would go here
        return {
            "status": "healthy",
            "agents": ["DataLoader", "Analytics", "Visualization", "Communication"],
            "mcp_server": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """
    Analyze data using the multi-agent system with MCP tools
    """
    try:
        session_id = request.session_id or "default"
        result = await run_multi_agent_analysis(request.query, session_id)
        
        return AnalysisResponse(
            result=result,
            session_id=session_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/agents")
async def get_agent_info():
    """Get information about available agents"""
    return {
        "agents": {
            "DataLoaderAgent": "File operations and data preparation",
            "AnalyticsAgent": "Statistical calculations and analysis", 
            "VisualizationAgent": "Chart creation and visual insights",
            "CommunicationAgent": "User-friendly response formatting"
        },
        "tools": [
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
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7000"))
    
    print(f"🌍 Starting Multi-Agent Backend on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
