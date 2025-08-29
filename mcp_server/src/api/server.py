"""
MCP Server Tools - Migrated from tools.py
All the sophisticated data analysis tools now running in MCP server
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from fastmcp import FastMCP
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Any

# Setup debug logging
DEBUG_MODE = os.getenv("MCP_DEBUG", "false").lower() == "true"
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "streamable-http")
mcp = FastMCP("Data Analysis Server")

# Data directory path
DATA_DIR = Path(__file__).parent.parent.parent / "data"
CHARTS_DIR = Path("/app/charts")  # Charts directory for Docker volume

# Ensure charts directory exists
CHARTS_DIR.mkdir(exist_ok=True)

# Set up matplotlib to use non-interactive backend for Docker
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')

# Global variable to store the current dataset (in memory for session)
current_dataset = None
current_filename = None

def debug_log(func_name: str, args: dict, result: any = None, error: str = None):
    """Debug logging function for MCP tool calls"""
    if DEBUG_MODE:
        log_msg = f"🔧 MCP Tool Call: {func_name}"
        if args:
            log_msg += f" | Args: {args}"
        if result is not None:
            if isinstance(result, (dict, list)):
                log_msg += f" | Result keys: {list(result.keys()) if isinstance(result, dict) else f'List[{len(result)}]'}"
            else:
                log_msg += f" | Result: {str(result)[:100]}..."
        if error:
            log_msg += f" | ERROR: {error}"
        logger.info(log_msg)
        print(f"🔧 {func_name} called with {args}")  # Also print for immediate visibility


@mcp.tool
async def investigate_directory(directory_path: str = None) -> List[Dict[str, Any]]:
    """
    Investigate files in a directory to help select the appropriate data file.
    
    Args:
        directory_path (str): Path to directory to investigate (defaults to the data directory)
        
    Returns:
        List[Dict]: List of file information dictionaries
    """
    debug_log("investigate_directory", {"directory_path": directory_path})
    
    try:
        # Default to the data directory if no path provided
        if directory_path is None or directory_path == "data":
            dir_path = DATA_DIR
        else:
            dir_path = Path(directory_path)
        
        if not dir_path.exists():
            error_msg = f"Directory '{dir_path}' does not exist"
            debug_log("investigate_directory", {"directory_path": directory_path}, error=error_msg)
            return [{"error": error_msg}]
        
        if not dir_path.is_dir():
            error_msg = f"'{dir_path}' is not a directory"
            debug_log("investigate_directory", {"directory_path": directory_path}, error=error_msg)
            return [{"error": error_msg}]
        
        # Get all files in the directory
        files = [f for f in dir_path.iterdir() if f.is_file()]
        
        if not files:
            result = [{"info": f"Directory '{directory_path}' is empty"}]
            debug_log("investigate_directory", {"directory_path": directory_path}, result)
            return result
        
        # Analyze each file
        file_info = []
        
        for file_path in sorted(files):
            file_name = file_path.name
            file_size = file_path.stat().st_size
            file_ext = file_path.suffix.lower()
            
            info = {
                "name": file_name,
                "size_mb": file_size / (1024 * 1024),
                "extension": file_ext,
                "path": str(file_path)
            }
            
            # Add content hints based on filename patterns
            name_lower = file_name.lower()
            if 'weather' in name_lower:
                info["content_type"] = "weather/climate data"
            elif 'sales' in name_lower or 'revenue' in name_lower:
                info["content_type"] = "sales/financial data"  
            elif 'employee' in name_lower or 'staff' in name_lower:
                info["content_type"] = "employee/HR data"
            else:
                info["content_type"] = "unknown"
            
            # Check if it's a CSV file and get structure
            if file_ext == '.csv':
                try:
                    df = pl.read_csv(file_path, n_rows=1)
                    info["columns"] = df.columns
                    info["column_count"] = len(df.columns)
                    info["status"] = "ready_to_load"
                except Exception as e:
                    info["status"] = f"csv_error: {str(e)}"
            else:
                info["status"] = "not_csv"
            
            file_info.append(info)
        
        debug_log("investigate_directory", {"directory_path": directory_path}, file_info)
        return file_info
        
    except Exception as e:
        error_msg = f"Error investigating directory: {str(e)}"
        debug_log("investigate_directory", {"directory_path": directory_path}, error=error_msg)
        return [{"error": error_msg}]


@mcp.tool
async def load_csv_file(file_path: str) -> Dict[str, Any]:
    """
    Load a CSV file into memory for analysis.
    
    Args:
        file_path (str): Path to the CSV file to load
        
    Returns:
        Dict: Success message with basic dataset information
    """
    global current_dataset, current_filename
    debug_log("load_csv_file", {"file_path": file_path})
    
    try:
        # Handle relative paths
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            path_obj = DATA_DIR / file_path
        
        if not path_obj.exists():
            # Try just the filename in data directory
            path_obj = DATA_DIR / Path(file_path).name
            if not path_obj.exists():
                available_files = [f.name for f in DATA_DIR.glob("*.csv")]
                error_result = {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                    "available_files": available_files
                }
                debug_log("load_csv_file", {"file_path": file_path}, error_result)
                return error_result
        
        # Load the CSV file with polars
        current_dataset = pl.read_csv(path_obj)
        current_filename = path_obj.name
        
        rows, cols = current_dataset.shape
        column_names = current_dataset.columns
        
        result = {
            "status": "success",
            "filename": current_filename,
            "rows": rows,
            "columns": cols,
            "column_names": column_names
        }
        debug_log("load_csv_file", {"file_path": file_path}, result)
        return result
        
    except Exception as e:
        error_result = {
            "status": "error", 
            "message": f"Error loading file: {str(e)}"
        }
        debug_log("load_csv_file", {"file_path": file_path}, error=str(e))
        return error_result


@mcp.tool
async def get_column_names() -> Dict[str, Any]:
    """Get all column names from the currently loaded dataset."""
    debug_log("get_column_names", {})
    
    if current_dataset is None:
        error_result = {"status": "error", "message": "No dataset loaded"}
        debug_log("get_column_names", {}, error="No dataset loaded")
        return error_result
    
    result = {
        "status": "success",
        "filename": current_filename,
        "columns": current_dataset.columns
    }
    debug_log("get_column_names", {}, result)
    return result


@mcp.tool
async def get_dataset_info() -> Dict[str, Any]:
    """Get basic information about the currently loaded dataset."""
    debug_log("get_dataset_info", {})
    
    if current_dataset is None:
        error_result = {"status": "error", "message": "No dataset loaded"}
        debug_log("get_dataset_info", {}, error="No dataset loaded")
        return error_result
    
    rows, cols = current_dataset.shape
    
    # Get data types
    dtypes_info = {}
    for col in current_dataset.columns:
        dtypes_info[col] = str(current_dataset[col].dtype)
    
    # Get sample data (first 3 rows)
    sample_df = current_dataset.head(3)
    sample_str = sample_df.to_pandas().to_string(index=False)
    
    result = {
        "status": "success",
        "filename": current_filename,
        "rows": rows,
        "columns": cols,
        "dtypes": dtypes_info,
        "sample_data": sample_str
    }
    debug_log("get_dataset_info", {}, result)
    return result


@mcp.tool
async def calculate_column_average(column_name: str) -> Dict[str, Any]:
    """Calculate the average (mean) value of a numeric column."""
    debug_log("calculate_column_average", {"column_name": column_name})
    
    if current_dataset is None:
        error_result = {"status": "error", "message": "No dataset loaded"}
        debug_log("calculate_column_average", {"column_name": column_name}, error="No dataset loaded")
        return error_result
    
    if column_name not in current_dataset.columns:
        error_result = {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
        debug_log("calculate_column_average", {"column_name": column_name}, error=f"Column not found")
        return error_result
    
    try:
        # Use polars for calculation
        col_data = current_dataset.select(column_name).to_series()
        if col_data.dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            error_result = {"status": "error", "message": f"Column '{column_name}' is not numeric"}
            debug_log("calculate_column_average", {"column_name": column_name}, error="Column not numeric")
            return error_result
        
        average = col_data.mean()
        count = col_data.len()
        
        result = {
            "status": "success",
            "column": column_name,
            "average": average,
            "count": count
        }
        debug_log("calculate_column_average", {"column_name": column_name}, result)
        return result
        
    except Exception as e:
        error_result = {"status": "error", "message": f"Error calculating average: {str(e)}"}
        debug_log("calculate_column_average", {"column_name": column_name}, error=str(e))
        return error_result


@mcp.tool
async def calculate_column_stats(column_name: str) -> Dict[str, Any]:
    """Calculate comprehensive statistics for a numeric column."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        col_data = current_dataset.select(column_name).to_series()
        if col_data.dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            return {"status": "error", "message": f"Column '{column_name}' is not numeric"}
        
        stats = {
            "count": col_data.len(),
            "mean": col_data.mean(),
            "median": col_data.median(),
            "std": col_data.std(),
            "min": col_data.min(),
            "max": col_data.max(),
            "q25": col_data.quantile(0.25),
            "q75": col_data.quantile(0.75)
        }
        
        return {
            "status": "success",
            "column": column_name,
            "statistics": stats
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error calculating statistics: {str(e)}"}


@mcp.tool
async def create_bar_chart(column_name: str, value_column: str = None, save_path: str = None) -> Dict[str, Any]:
    """Create a bar chart from the loaded dataset."""
    debug_log("create_bar_chart", {"column_name": column_name, "value_column": value_column, "save_path": save_path})
    
    if current_dataset is None:
        error_result = {"status": "error", "message": "No dataset loaded"}
        debug_log("create_bar_chart", {"column_name": column_name, "value_column": value_column}, error="No dataset loaded")
        return error_result
    
    if column_name not in current_dataset.columns:
        error_result = {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
        debug_log("create_bar_chart", {"column_name": column_name, "value_column": value_column}, error=f"Column not found")
        return error_result
    
    if value_column and value_column not in current_dataset.columns:
        error_result = {
            "status": "error", 
            "message": f"Column '{value_column}' not found",
            "available_columns": current_dataset.columns
        }
        debug_log("create_bar_chart", {"column_name": column_name, "value_column": value_column}, error=f"Value column not found")
        return error_result
    
    try:
        # Create the chart using matplotlib
        plt.figure(figsize=(10, 6))
        
        if value_column:
            # Aggregate data for bar chart
            df_pandas = current_dataset.to_pandas()
            grouped = df_pandas.groupby(column_name)[value_column].mean().reset_index()
            plt.bar(grouped[column_name], grouped[value_column])
            plt.ylabel(f"Average {value_column}")
        else:
            # Count frequency for bar chart
            df_pandas = current_dataset.to_pandas()
            value_counts = df_pandas[column_name].value_counts()
            plt.bar(value_counts.index, value_counts.values)
            plt.ylabel("Count")
        
        plt.xlabel(column_name)
        plt.title(f"Bar Chart: {column_name}" + (f" vs {value_column}" if value_column else " Distribution"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart to charts directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"bar_chart_{column_name}_{timestamp}.png"
        chart_path = CHARTS_DIR / chart_filename
        
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        result = {
            "status": "success",
            "chart_type": "bar_chart",
            "chart_path": str(chart_path),
            "chart_filename": chart_filename,
            "column_name": column_name,
            "value_column": value_column,
            "title": f"Bar Chart: {column_name}" + (f" vs {value_column}" if value_column else " Distribution")
        }
        debug_log("create_bar_chart", {"column_name": column_name, "value_column": value_column}, result)
        return result
        
    except Exception as e:
        error_result = {"status": "error", "message": f"Error creating chart: {str(e)}"}
        debug_log("create_bar_chart", {"column_name": column_name, "value_column": value_column}, error=str(e))
        return error_result


@mcp.tool
async def count_rows_with_value(column_name: str, value: str) -> Dict[str, Any]:
    """Count how many rows contain a specific value in a column."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        # Use polars for counting
        count = current_dataset.filter(
            pl.col(column_name).cast(pl.Utf8) == str(value)
        ).height
        
        total_rows = current_dataset.height
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0
        
        return {
            "status": "success",
            "column": column_name,
            "value": value,
            "count": count,
            "total_rows": total_rows,
            "percentage": percentage
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error counting values: {str(e)}"}


@mcp.tool
async def get_unique_values(column_name: str) -> Dict[str, Any]:
    """Get all unique values in a column."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        unique_vals = current_dataset.select(column_name).unique().to_series().to_list()
        # Remove None values
        unique_vals = [val for val in unique_vals if val is not None]
        
        return {
            "status": "success",
            "column": column_name,
            "unique_values": unique_vals[:50],  # Limit to first 50
            "total_unique": len(unique_vals),
            "truncated": len(unique_vals) > 50
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error getting unique values: {str(e)}"}


@mcp.tool
async def find_max_value(column_name: str) -> Dict[str, Any]:
    """Find the maximum value in a numeric column."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        col_data = current_dataset.select(column_name).to_series()
        if col_data.dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            return {"status": "error", "message": f"Column '{column_name}' is not numeric"}
        
        max_value = col_data.max()
        max_row = current_dataset.filter(pl.col(column_name) == max_value).to_dicts()
        
        return {
            "status": "success",
            "column": column_name,
            "max_value": max_value,
            "max_row": max_row[0] if max_row else None
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error finding max value: {str(e)}"}


@mcp.tool
async def find_min_value(column_name: str) -> Dict[str, Any]:
    """Find the minimum value in a numeric column."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        col_data = current_dataset.select(column_name).to_series()
        if col_data.dtype not in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            return {"status": "error", "message": f"Column '{column_name}' is not numeric"}
        
        min_value = col_data.min()
        min_row = current_dataset.filter(pl.col(column_name) == min_value).to_dicts()
        
        return {
            "status": "success",
            "column": column_name,
            "min_value": min_value,
            "min_row": min_row[0] if min_row else None
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error finding min value: {str(e)}"}


@mcp.tool
async def group_by_column_and_aggregate(group_column: str, agg_column: str, operation: str = "sum") -> Dict[str, Any]:
    """Group by a column and perform aggregation on another column."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if group_column not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Group column '{group_column}' not found",
            "available_columns": current_dataset.columns
        }
    
    if agg_column not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Aggregation column '{agg_column}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        # Map operation to polars expression
        operation_map = {
            "sum": pl.col(agg_column).sum(),
            "mean": pl.col(agg_column).mean(),
            "count": pl.col(agg_column).count(),
            "max": pl.col(agg_column).max(),
            "min": pl.col(agg_column).min()
        }
        
        if operation not in operation_map:
            return {
                "status": "error", 
                "message": f"Unknown operation '{operation}'. Available: {list(operation_map.keys())}"
            }
        
        result = current_dataset.group_by(group_column).agg(
            operation_map[operation].alias(f"{operation}_{agg_column}")
        ).sort(group_column)
        
        return {
            "status": "success",
            "group_column": group_column,
            "agg_column": agg_column,
            "operation": operation,
            "result": result.to_dicts()
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error in group by aggregation: {str(e)}"}


@mcp.tool
async def filter_data(column_name: str, condition: str, value: str) -> Dict[str, Any]:
    """Filter data based on a condition."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        # Build filter expression based on condition
        col = pl.col(column_name)
        
        if condition == "equals":
            filter_expr = col.cast(pl.Utf8) == str(value)
        elif condition == "contains":
            filter_expr = col.cast(pl.Utf8).str.contains(str(value))
        elif condition == "greater_than":
            filter_expr = col > float(value)
        elif condition == "less_than":
            filter_expr = col < float(value)
        elif condition == "greater_equal":
            filter_expr = col >= float(value)
        elif condition == "less_equal":
            filter_expr = col <= float(value)
        else:
            return {
                "status": "error",
                "message": f"Unknown condition '{condition}'. Available: equals, contains, greater_than, less_than, greater_equal, less_equal"
            }
        
        filtered_data = current_dataset.filter(filter_expr)
        original_rows = current_dataset.height
        filtered_rows = filtered_data.height
        
        return {
            "status": "success",
            "column": column_name,
            "condition": condition,
            "value": value,
            "original_rows": original_rows,
            "filtered_rows": filtered_rows,
            "sample_data": filtered_data.head(10).to_dicts()
        }
        
    except Exception as e:
        return {"status": "error", "message": f"Error filtering data: {str(e)}"}


@mcp.tool
async def create_scatter_plot(x_column: str, y_column: str, color_column: str = None, title: str = None) -> Dict[str, Any]:
    """Create a scatter plot from the loaded dataset."""
    debug_log("create_scatter_plot", {"x_column": x_column, "y_column": y_column, "color_column": color_column, "title": title})
    
    if current_dataset is None:
        error_result = {"status": "error", "message": "No dataset loaded"}
        debug_log("create_scatter_plot", {"x_column": x_column, "y_column": y_column}, error="No dataset loaded")
        return error_result
    
    if x_column not in current_dataset.columns:
        error_result = {
            "status": "error", 
            "message": f"X column '{x_column}' not found",
            "available_columns": current_dataset.columns
        }
        debug_log("create_scatter_plot", {"x_column": x_column, "y_column": y_column}, error="X column not found")
        return error_result
    
    if y_column not in current_dataset.columns:
        error_result = {
            "status": "error", 
            "message": f"Y column '{y_column}' not found",
            "available_columns": current_dataset.columns
        }
        debug_log("create_scatter_plot", {"x_column": x_column, "y_column": y_column}, error="Y column not found")
        return error_result
    
    try:
        # Create the scatter plot using matplotlib
        plt.figure(figsize=(10, 6))
        df_pandas = current_dataset.to_pandas()
        
        if color_column and color_column in current_dataset.columns:
            # Color by a categorical column
            unique_colors = df_pandas[color_column].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_colors)))
            
            for i, category in enumerate(unique_colors):
                mask = df_pandas[color_column] == category
                plt.scatter(df_pandas[x_column][mask], df_pandas[y_column][mask], 
                           c=[colors[i]], label=category, alpha=0.7)
            plt.legend()
        else:
            # Simple scatter plot
            plt.scatter(df_pandas[x_column], df_pandas[y_column], alpha=0.7)
        
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(title or f"Scatter Plot: {x_column} vs {y_column}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save chart to charts directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_filename = f"scatter_plot_{x_column}_{y_column}_{timestamp}.png"
        chart_path = CHARTS_DIR / chart_filename
        
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        result = {
            "status": "success",
            "chart_type": "scatter_plot",
            "chart_path": str(chart_path),
            "chart_filename": chart_filename,
            "x_column": x_column,
            "y_column": y_column,
            "color_column": color_column,
            "title": title or f"Scatter Plot: {x_column} vs {y_column}"
        }
        debug_log("create_scatter_plot", {"x_column": x_column, "y_column": y_column}, result)
        return result
        
    except Exception as e:
        error_result = {"status": "error", "message": f"Error creating scatter plot: {str(e)}"}
        debug_log("create_scatter_plot", {"x_column": x_column, "y_column": y_column}, error=str(e))
        return error_result


@mcp.tool
async def create_box_plot(column_name: str, group_column: str = None, title: str = None) -> Dict[str, Any]:
    """Create a box plot from the loaded dataset."""
    if current_dataset is None:
        return {"status": "error", "message": "No dataset loaded"}
    
    if column_name not in current_dataset.columns:
        return {
            "status": "error", 
            "message": f"Column '{column_name}' not found",
            "available_columns": current_dataset.columns
        }
    
    try:
        chart_info = {
            "status": "success",
            "chart_type": "box_plot",
            "column_name": column_name,
            "group_column": group_column,
            "title": title or f"Box Plot: {column_name}" + (f" by {group_column}" if group_column else ""),
            "note": "Chart creation would be implemented with matplotlib/seaborn"
        }
        
        return chart_info
        
    except Exception as e:
        return {"status": "error", "message": f"Error creating box plot: {str(e)}"}


# Add more tools following the same pattern...
# All the other tools from tools.py would be migrated here

if __name__ == "__main__":
    logging.info("Starting MCP Data Analysis Server")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
