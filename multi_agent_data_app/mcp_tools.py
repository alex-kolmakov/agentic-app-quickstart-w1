"""
MCP Client Tools for Agents
This module provides synchronous MCP client tools that agents can use to communicate with the MCP server.
These tools replace the local tools and provide debugging capabilities.
"""

import requests
import json
import os
from typing import Any, Dict
from agents import function_tool
import logging

logger = logging.getLogger(__name__)

# MCP Server configuration
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
DEBUG_MODE = os.getenv("MCP_DEBUG", "false").lower() == "true"

def debug_print(tool_name: str, args: dict, result: any = None, error: str = None):
    """Debug print function for MCP client calls"""
    if DEBUG_MODE:
        print(f"🔗 MCP Client: {tool_name} -> {MCP_SERVER_URL}")
        if args:
            print(f"   📤 Args: {args}")
        if result is not None:
            if isinstance(result, (dict, list)):
                print(f"   📥 Result: {type(result).__name__} with {len(result)} items")
            else:
                print(f"   📥 Result: {str(result)[:200]}...")
        if error:
            print(f"   ❌ Error: {error}")

def call_mcp_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Call a tool on the MCP server using HTTP requests"""
    url = f"{MCP_SERVER_URL}/mcp/tools/{tool_name}"
    debug_print(tool_name, kwargs)
    
    try:
        # Make HTTP POST request to MCP server
        response = requests.post(
            url, 
            json=kwargs,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = response.json()
        debug_print(tool_name, kwargs, result)
        return result
    except requests.exceptions.RequestException as e:
        error_msg = f"MCP HTTP call failed: {str(e)}"
        debug_print(tool_name, kwargs, error=error_msg)
        return {"status": "error", "message": error_msg}
    except Exception as e:
        error_msg = f"MCP call failed: {str(e)}"
        debug_print(tool_name, kwargs, error=error_msg)
        return {"status": "error", "message": error_msg}

# =============================================================================
# MCP CLIENT TOOLS - These replace the local tools
# =============================================================================

@function_tool
def investigate_directory(directory_path: str = None) -> str:
    """
    Investigate files in a directory via MCP server to help select the appropriate data file.
    
    Args:
        directory_path (str): Path to directory to investigate (defaults to the data directory)
        
    Returns:
        str: List of files with descriptions and recommendations
    """
    result = call_mcp_tool("investigate_directory", directory_path=directory_path)
    
    # Format result for agent consumption
    if isinstance(result, list):
        if result and "error" in result[0]:
            return f"❌ {result[0]['error']}"
        
        # Format file information nicely
        formatted = f"🔍 Found {len(result)} files:\n\n"
        csv_count = 0
        
        for file_info in result:
            if "error" in file_info:
                formatted += f"❌ {file_info['error']}\n"
                continue
                
            name = file_info.get("name", "unknown")
            size_mb = file_info.get("size_mb", 0)
            content_type = file_info.get("content_type", "unknown")
            status = file_info.get("status", "unknown")
            columns = file_info.get("columns", [])
            
            formatted += f"📄 {name} ({size_mb:.2f} MB)"
            
            if content_type != "unknown":
                if "weather" in content_type:
                    formatted += " - 🌤️ Weather/climate data"
                elif "sales" in content_type:
                    formatted += " - 💰 Sales/financial data"
                elif "employee" in content_type:
                    formatted += " - 👥 Employee/HR data"
            
            if status == "ready_to_load":
                formatted += " ✅ Ready to load"
                csv_count += 1
                if columns and len(columns) <= 8:
                    formatted += f" (Cols: {', '.join(columns)})"
            
            formatted += "\n"
        
        if csv_count > 0:
            formatted += f"\n💡 Found {csv_count} CSV files ready to load with load_csv_file()"
        
        return formatted
    
    elif isinstance(result, dict) and "error" in result:
        return f"❌ {result['error']}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def load_csv_file(file_path: str) -> str:
    """
    Load a CSV file via MCP server into memory for analysis.
    
    Args:
        file_path (str): Path to the CSV file to load
        
    Returns:
        str: Success message with basic dataset information
    """
    result = call_mcp_tool("load_csv_file", file_path=file_path)
    
    # Format result for agent consumption
    if isinstance(result, dict):
        if result.get("status") == "success":
            filename = result.get("filename", "unknown")
            rows = result.get("rows", 0)
            cols = result.get("columns", 0)
            columns = result.get("column_names", [])
            
            return f"✅ Successfully loaded '{filename}' with {rows} rows and {cols} columns.\nColumns: {', '.join(columns)}"
        
        elif result.get("status") == "error":
            message = result.get("message", "Unknown error")
            available = result.get("available_files", [])
            
            error_msg = f"❌ Error: {message}"
            if available:
                error_msg += f"\n📁 Available CSV files: {', '.join(available)}"
            return error_msg
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def get_column_names() -> str:
    """
    Get all column names from the currently loaded dataset via MCP server.
    
    Returns:
        str: List of column names or error message
    """
    result = call_mcp_tool("get_column_names")
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            columns = result.get("columns", [])
            filename = result.get("filename", "dataset")
            return f"Column names in '{filename}': {', '.join(columns)}"
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def get_dataset_info() -> str:
    """
    Get basic information about the currently loaded dataset via MCP server.
    
    Returns:
        str: Dataset summary including shape, data types, and sample data
    """
    result = call_mcp_tool("get_dataset_info")
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            filename = result.get("filename", "dataset")
            rows = result.get("rows", 0)
            cols = result.get("columns", 0)
            dtypes = result.get("dtypes", {})
            sample = result.get("sample_data", "No sample available")
            
            dtypes_info = []
            for col, dtype in dtypes.items():
                dtypes_info.append(f"  {col}: {dtype}")
            
            return f"""📊 Dataset Information for '{filename}':
- Shape: {rows} rows × {cols} columns
- Data types:
{chr(10).join(dtypes_info)}

📋 Sample data:
{sample}"""
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def calculate_column_average(column_name: str) -> str:
    """
    Calculate the average (mean) value of a numeric column via MCP server.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Average value or error message
    """
    result = call_mcp_tool("calculate_column_average", column_name=column_name)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            average = result.get("average", 0)
            count = result.get("count", 0)
            return f"📊 Average of '{column_name}': {average:.2f} (based on {count} numeric values)"
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def calculate_column_stats(column_name: str) -> str:
    """
    Calculate comprehensive statistics for a numeric column via MCP server.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Detailed statistics including mean, median, std, min, max, etc.
    """
    result = call_mcp_tool("calculate_column_stats", column_name=column_name)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            stats = result.get("statistics", {})
            count = result.get("count", 0)
            
            formatted = f"📊 Statistics for '{column_name}' ({count} values):\n"
            for stat_name, value in stats.items():
                if isinstance(value, float):
                    formatted += f"• {stat_name.title()}: {value:.2f}\n"
                else:
                    formatted += f"• {stat_name.title()}: {value}\n"
            
            return formatted.rstrip()
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def create_bar_chart(column_name: str, value_column: str = None, save_path: str = None) -> str:
    """
    Create a bar chart via MCP server.
    
    Args:
        column_name (str): Categorical column for x-axis
        value_column (str): Optional numeric column for y-axis (if not provided, shows counts)
        save_path (str): Optional path to save the chart image
        
    Returns:
        str: Success message with chart information
    """
    result = call_mcp_tool("create_bar_chart", 
                          column_name=column_name, 
                          value_column=value_column, 
                          save_path=save_path)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            chart_path = result.get("chart_path", "unknown")
            chart_filename = result.get("chart_filename", "unknown")
            return f"📊 Bar chart created successfully: {chart_filename}\nSaved to: {chart_path}"
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def create_scatter_plot(x_column: str, y_column: str, color_column: str = None, save_path: str = None) -> str:
    """
    Create a scatter plot via MCP server.
    
    Args:
        x_column (str): Column for x-axis
        y_column (str): Column for y-axis
        color_column (str): Column for color coding - optional
        save_path (str): Optional path to save the chart image
        
    Returns:
        str: Success message with chart information
    """
    result = call_mcp_tool("create_scatter_plot", 
                          x_column=x_column, 
                          y_column=y_column,
                          color_column=color_column,
                          save_path=save_path)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            chart_path = result.get("chart_path", "unknown")
            chart_filename = result.get("chart_filename", "unknown")
            return f"🔍 Scatter plot created successfully: {chart_filename}\nSaved to: {chart_path}"
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def create_box_plot(column_name: str, group_column: str = None, save_path: str = None) -> str:
    """
    Create a box plot via MCP server.
    
    Args:
        column_name (str): Column to analyze for outliers
        group_column (str): Column to group by - optional
        save_path (str): Optional path to save the chart image
        
    Returns:
        str: Success message with chart information
    """
    result = call_mcp_tool("create_box_plot", 
                          column_name=column_name,
                          group_column=group_column,
                          save_path=save_path)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            chart_path = result.get("chart_path", "unknown")
            chart_filename = result.get("chart_filename", "unknown")
            return f"📦 Box plot created successfully: {chart_filename}\nSaved to: {chart_path}"
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


# Additional MCP tools

@function_tool
def count_rows_with_value(column_name: str, value: str) -> str:
    """Count rows with specific value via MCP server."""
    result = call_mcp_tool("count_rows_with_value", column_name=column_name, value=value)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            count = result.get("count", 0)
            return f"📊 Found {count} rows where '{column_name}' = '{value}'"
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def get_unique_values(column_name: str) -> str:
    """Get unique values from a column via MCP server."""
    result = call_mcp_tool("get_unique_values", column_name=column_name)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            values = result.get("unique_values", [])
            count = result.get("count", 0)
            
            if count <= 20:  # Show all values if not too many
                return f"🎯 Unique values in '{column_name}' ({count} total): {', '.join(map(str, values))}"
            else:
                return f"🎯 Unique values in '{column_name}' ({count} total): {', '.join(map(str, values[:10]))} ... and {count-10} more"
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def find_max_value(column_name: str) -> str:
    """Find maximum value in a column via MCP server."""
    result = call_mcp_tool("find_max_value", column_name=column_name)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            max_val = result.get("max_value")
            return f"📈 Maximum value in '{column_name}': {max_val}"
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def find_min_value(column_name: str) -> str:
    """Find minimum value in a column via MCP server."""
    result = call_mcp_tool("find_min_value", column_name=column_name)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            min_val = result.get("min_value")
            return f"📉 Minimum value in '{column_name}': {min_val}"
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def group_by_column_and_aggregate(group_column: str, agg_column: str, operation: str = "sum") -> str:
    """Group data and aggregate via MCP server."""
    result = call_mcp_tool("group_by_column_and_aggregate", 
                          group_column=group_column,
                          agg_column=agg_column,
                          operation=operation)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            grouped_data = result.get("grouped_data", {})
            formatted = f"📊 {operation.title()} of '{agg_column}' by '{group_column}':\n"
            for group, value in grouped_data.items():
                if isinstance(value, float):
                    formatted += f"• {group}: {value:.2f}\n"
                else:
                    formatted += f"• {group}: {value}\n"
            return formatted.rstrip()
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


@function_tool
def filter_data(column_name: str, condition: str, value: str) -> str:
    """Filter data based on condition via MCP server."""
    result = call_mcp_tool("filter_data", 
                          column_name=column_name,
                          condition=condition,
                          value=value)
    
    if isinstance(result, dict):
        if result.get("status") == "success":
            filtered_count = result.get("filtered_count", 0)
            total_count = result.get("total_count", 0)
            sample_data = result.get("sample_data", "No sample available")
            
            return f"🔍 Filtered {filtered_count} rows out of {total_count} where '{column_name}' {condition} '{value}'\n\nSample filtered data:\n{sample_data}"
        
        elif result.get("status") == "error":
            return f"❌ {result.get('message', 'Unknown error')}"
    
    return f"❌ Unexpected result format: {result}"


# Export all MCP client tools for agents to use
MCP_TOOLS = [
    investigate_directory,
    load_csv_file, 
    get_column_names,
    get_dataset_info,
    calculate_column_average,
    calculate_column_stats,
    create_bar_chart,
    create_scatter_plot,
    create_box_plot,
    count_rows_with_value,
    get_unique_values,
    find_max_value,
    find_min_value,
    group_by_column_and_aggregate,
    filter_data
]
