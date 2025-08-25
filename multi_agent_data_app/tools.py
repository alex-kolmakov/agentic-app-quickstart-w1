"""
Data Analysis Tools for CSV Processing with Visualization

This module provides a comprehensive set of tools for analyzing CSV data and creating visualizations.
These tools can be used by agents to perform various data operations, answer user questions,
and generate insightful charts about their datasets.

Key Features:
- CSV file loading and validation
- Statistical calculations (mean, median, mode, etc.)
- Data filtering and aggregation
- Column analysis and metadata extraction
- 📈 Data visualization with matplotlib and seaborn
- Chart export functionality
- Error handling for robust operations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agents import function_tool
import os
from pathlib import Path
import numpy as np
from datetime import datetime


# Global variable to store the current dataset
current_dataset = None
current_filename = None


@function_tool
def investigate_directory(directory_path: str = None) -> str:
    """
    Investigate files in a directory to help select the appropriate data file.
    
    Args:
        directory_path (str): Path to directory to investigate (defaults to ./data)
        
    Returns:
        str: List of files with descriptions and recommendations
    """
    try:
        # Default to the data directory if no path provided
        if directory_path is None:
            data_dir = Path(__file__).parent / "data"
            directory_path = str(data_dir)
        
        # Convert to Path object for easier handling
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            return f"❌ Directory '{directory_path}' does not exist."
        
        if not dir_path.is_dir():
            return f"❌ '{directory_path}' is not a directory."
        
        # Get all files in the directory
        files = [f for f in dir_path.iterdir() if f.is_file()]
        
        if not files:
            return f"📂 Directory '{directory_path}' is empty."
        
        # Analyze each file
        file_info = []
        csv_files = []
        
        for file_path in sorted(files):
            file_name = file_path.name
            file_size = file_path.stat().st_size
            file_ext = file_path.suffix.lower()
            
            # Basic file info
            size_mb = file_size / (1024 * 1024)
            info = f"📄 {file_name} ({size_mb:.2f} MB)"
            
            # Add content hints based on filename patterns
            name_lower = file_name.lower()
            if 'weather' in name_lower:
                info += " - 🌤️ Likely contains weather/climate data"
            elif 'sales' in name_lower or 'revenue' in name_lower:
                info += " - 💰 Likely contains sales/financial data"  
            elif 'employee' in name_lower or 'staff' in name_lower or 'personnel' in name_lower:
                info += " - 👥 Likely contains employee/HR data"
            elif 'customer' in name_lower or 'client' in name_lower:
                info += " - 🤝 Likely contains customer data"
            elif 'product' in name_lower or 'inventory' in name_lower:
                info += " - 📦 Likely contains product/inventory data"
            elif 'transaction' in name_lower or 'order' in name_lower:
                info += " - 🛒 Likely contains transaction/order data"
            
            # Check if it's a CSV file
            if file_ext == '.csv':
                csv_files.append(file_path)
                info += " ✅ CSV format - ready to load"
                
                # Try to peek at CSV structure without loading fully
                try:
                    sample_df = pd.read_csv(file_path, nrows=0)  # Just get column names
                    cols = list(sample_df.columns)
                    info += f" (Columns: {len(cols)})"
                    if len(cols) <= 8:  # Show column names if not too many
                        info += f" [{', '.join(cols)}]"
                except Exception:
                    info += " (Unable to read CSV structure)"
            elif file_ext in ['.xlsx', '.xls']:
                info += " 📊 Excel format - may need conversion"
            elif file_ext in ['.json']:
                info += " 🔗 JSON format - may need conversion"
            elif file_ext in ['.txt', '.log']:
                info += " 📝 Text format - may need parsing"
            else:
                info += f" ❓ {file_ext.upper()} format"
            
            file_info.append(info)
        
        # Build response
        result = f"🔍 Found {len(files)} files in '{directory_path}':\n\n"
        result += "\n".join(file_info)
        
        # Add recommendations
        if csv_files:
            result += f"\n\n💡 Recommendations:"
            result += f"\n• {len(csv_files)} CSV files are ready to load with load_csv_file()"
            
            # Provide specific suggestions based on common requests
            result += f"\n• For weather data: Look for files with 'weather' in the name"
            result += f"\n• For sales data: Look for files with 'sales' or 'revenue' in the name"
            result += f"\n• For employee data: Look for files with 'employee', 'staff', or 'personnel' in the name"
        
        return result
        
    except Exception as e:
        return f"❌ Error investigating directory: {str(e)}"


@function_tool
def load_csv_file(file_path: str) -> str:
    """
    Load a CSV file into memory for analysis.
    
    Args:
        file_path (str): Path to the CSV file to load
        
    Returns:
        str: Success message with basic dataset information
    """
    global current_dataset, current_filename
    
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(file_path):
            # Look in the data directory first
            data_dir = Path(__file__).parent / "data"
            potential_path = data_dir / file_path
            if potential_path.exists():
                file_path = str(potential_path)
        
        # Load the CSV file
        current_dataset = pd.read_csv(file_path)
        current_filename = os.path.basename(file_path)
        
        rows, cols = current_dataset.shape
        column_names = list(current_dataset.columns)
        
        return f"✅ Successfully loaded '{current_filename}' with {rows} rows and {cols} columns.\nColumns: {', '.join(column_names)}"
        
    except FileNotFoundError:
        return f"❌ Error: File '{file_path}' not found. Please check the file path."
    except Exception as e:
        return f"❌ Error loading file: {str(e)}"


@function_tool
def get_column_names() -> str:
    """
    Get all column names from the currently loaded dataset.
    
    Returns:
        str: List of column names or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first using load_csv_file()."
    
    columns = list(current_dataset.columns)
    return f"Column names in '{current_filename}': {', '.join(columns)}"


@function_tool
def get_dataset_info() -> str:
    """
    Get basic information about the currently loaded dataset.
    
    Returns:
        str: Dataset summary including shape, data types, and sample data
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    rows, cols = current_dataset.shape
    
    # Get data types
    dtypes_info = []
    for col, dtype in current_dataset.dtypes.items():
        dtypes_info.append(f"{col}: {dtype}")
    
    # Get first few rows as sample
    sample_data = current_dataset.head(3).to_string(index=False)
    
    return f"""📊 Dataset Information for '{current_filename}':
- Shape: {rows} rows × {cols} columns
- Data types:
  {chr(10).join(dtypes_info)}

📋 Sample data (first 3 rows):
{sample_data}"""


@function_tool
def calculate_column_average(column_name: str) -> str:
    """
    Calculate the average (mean) value of a numeric column.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Average value or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        column_data = pd.to_numeric(current_dataset[column_name], errors='coerce')
        if column_data.isna().all():
            return f"❌ Column '{column_name}' contains no numeric data."
        
        # Remove NaN values and calculate mean
        clean_data = column_data.dropna()
        average = clean_data.mean()
        
        return f"📊 Average of '{column_name}': {average:.2f} (based on {len(clean_data)} numeric values)"
        
    except Exception as e:
        return f"❌ Error calculating average: {str(e)}"


@function_tool
def calculate_column_stats(column_name: str) -> str:
    """
    Calculate comprehensive statistics for a numeric column.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Detailed statistics including mean, median, std, min, max, etc.
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        column_data = pd.to_numeric(current_dataset[column_name], errors='coerce')
        if column_data.isna().all():
            return f"❌ Column '{column_name}' contains no numeric data."
        
        clean_data = column_data.dropna()
        stats = clean_data.describe()
        
        return f"""📊 Statistics for '{column_name}':
- Count: {int(stats['count'])} values
- Mean: {stats['mean']:.2f}
- Median (50%): {stats['50%']:.2f}
- Standard Deviation: {stats['std']:.2f}
- Minimum: {stats['min']:.2f}
- Maximum: {stats['max']:.2f}
- 25th Percentile: {stats['25%']:.2f}
- 75th Percentile: {stats['75%']:.2f}"""
        
    except Exception as e:
        return f"❌ Error calculating statistics: {str(e)}"


@function_tool
def count_rows_with_value(column_name: str, value: str) -> str:
    """
    Count how many rows contain a specific value in a column.
    
    Args:
        column_name (str): Name of the column to search
        value (str): Value to count
        
    Returns:
        str: Count of matching rows or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        # Convert value to appropriate type for comparison
        column_data = current_dataset[column_name].astype(str)
        count = (column_data == str(value)).sum()
        total_rows = len(current_dataset)
        percentage = (count / total_rows) * 100 if total_rows > 0 else 0
        
        return f"🔍 Found {count} rows with '{value}' in column '{column_name}' ({percentage:.1f}% of {total_rows} total rows)"
        
    except Exception as e:
        return f"❌ Error counting values: {str(e)}"


@function_tool
def get_unique_values(column_name: str) -> str:
    """
    Get all unique values in a column.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: List of unique values or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        unique_vals = current_dataset[column_name].unique()
        # Remove NaN values and convert to string for display
        unique_vals = [str(val) for val in unique_vals if pd.notna(val)]
        
        if len(unique_vals) > 20:
            displayed_vals = unique_vals[:20]
            return f"🔍 Unique values in '{column_name}' (showing first 20 of {len(unique_vals)}): {', '.join(displayed_vals)}..."
        else:
            return f"🔍 Unique values in '{column_name}': {', '.join(unique_vals)}"
        
    except Exception as e:
        return f"❌ Error getting unique values: {str(e)}"


@function_tool
def find_max_value(column_name: str) -> str:
    """
    Find the maximum value in a numeric column.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Maximum value and corresponding row information
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        column_data = pd.to_numeric(current_dataset[column_name], errors='coerce')
        if column_data.isna().all():
            return f"❌ Column '{column_name}' contains no numeric data."
        
        max_value = column_data.max()
        max_index = column_data.idxmax()
        
        # Get the full row for context
        max_row = current_dataset.loc[max_index]
        row_info = ', '.join([f"{col}: {val}" for col, val in max_row.items()])
        
        return f"📈 Maximum value in '{column_name}': {max_value}\n📋 Full row: {row_info}"
        
    except Exception as e:
        return f"❌ Error finding maximum: {str(e)}"


@function_tool
def find_min_value(column_name: str) -> str:
    """
    Find the minimum value in a numeric column.
    
    Args:
        column_name (str): Name of the column to analyze
        
    Returns:
        str: Minimum value and corresponding row information
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        column_data = pd.to_numeric(current_dataset[column_name], errors='coerce')
        if column_data.isna().all():
            return f"❌ Column '{column_name}' contains no numeric data."
        
        min_value = column_data.min()
        min_index = column_data.idxmin()
        
        # Get the full row for context
        min_row = current_dataset.loc[min_index]
        row_info = ', '.join([f"{col}: {val}" for col, val in min_row.items()])
        
        return f"📉 Minimum value in '{column_name}': {min_value}\n📋 Full row: {row_info}"
        
    except Exception as e:
        return f"❌ Error finding minimum: {str(e)}"


@function_tool
def group_by_column_and_aggregate(group_column: str, agg_column: str, operation: str = "sum") -> str:
    """
    Group data by one column and aggregate another column.
    
    Args:
        group_column (str): Column to group by
        agg_column (str): Column to aggregate
        operation (str): Aggregation operation (sum, mean, count, min, max)
        
    Returns:
        str: Grouped results or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if group_column not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Group column '{group_column}' not found. Available columns: {available_cols}"
    
    if agg_column not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Aggregation column '{agg_column}' not found. Available columns: {available_cols}"
    
    valid_operations = ["sum", "mean", "count", "min", "max"]
    if operation not in valid_operations:
        return f"❌ Invalid operation '{operation}'. Valid operations: {', '.join(valid_operations)}"
    
    try:
        if operation == "count":
            # For count, we don't need numeric data
            result = current_dataset.groupby(group_column)[agg_column].count()
        else:
            # For other operations, convert to numeric
            numeric_data = pd.to_numeric(current_dataset[agg_column], errors='coerce')
            if numeric_data.isna().all():
                return f"❌ Column '{agg_column}' contains no numeric data for {operation} operation."
            
            # Create a temporary dataframe with cleaned data
            temp_df = current_dataset.copy()
            temp_df[agg_column] = numeric_data
            temp_df = temp_df.dropna(subset=[agg_column])
            
            if operation == "sum":
                result = temp_df.groupby(group_column)[agg_column].sum()
            elif operation == "mean":
                result = temp_df.groupby(group_column)[agg_column].mean()
            elif operation == "min":
                result = temp_df.groupby(group_column)[agg_column].min()
            elif operation == "max":
                result = temp_df.groupby(group_column)[agg_column].max()
        
        # Format results
        result_lines = []
        for group, value in result.items():
            if operation in ["sum", "mean", "min", "max"]:
                result_lines.append(f"{group}: {value:.2f}")
            else:
                result_lines.append(f"{group}: {value}")
        
        return f"📊 {operation.title()} of '{agg_column}' grouped by '{group_column}':\n" + "\n".join(result_lines)
        
    except Exception as e:
        return f"❌ Error in grouping operation: {str(e)}"


@function_tool
def filter_data(column_name: str, condition: str, value: str) -> str:
    """
    Filter the dataset based on a condition and show results.
    
    Args:
        column_name (str): Column to filter on
        condition (str): Condition type (equals, greater_than, less_than, contains)
        value (str): Value to compare against
        
    Returns:
        str: Filtered results or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    valid_conditions = ["equals", "greater_than", "less_than", "contains"]
    if condition not in valid_conditions:
        return f"❌ Invalid condition '{condition}'. Valid conditions: {', '.join(valid_conditions)}"
    
    try:
        if condition == "equals":
            filtered_df = current_dataset[current_dataset[column_name].astype(str) == str(value)]
        elif condition == "contains":
            filtered_df = current_dataset[current_dataset[column_name].astype(str).str.contains(str(value), case=False, na=False)]
        elif condition in ["greater_than", "less_than"]:
            # For numeric comparisons
            numeric_data = pd.to_numeric(current_dataset[column_name], errors='coerce')
            try:
                numeric_value = float(value)
            except ValueError:
                return f"❌ Cannot convert '{value}' to number for {condition} comparison."
            
            if condition == "greater_than":
                filtered_df = current_dataset[numeric_data > numeric_value]
            else:  # less_than
                filtered_df = current_dataset[numeric_data < numeric_value]
        
        if len(filtered_df) == 0:
            return f"🔍 No rows found where {column_name} {condition.replace('_', ' ')} '{value}'"
        
        # Show results summary
        result_count = len(filtered_df)
        total_count = len(current_dataset)
        
        # Show first few rows as sample
        if result_count <= 5:
            sample_data = filtered_df.to_string(index=False)
        else:
            sample_data = filtered_df.head(5).to_string(index=False)
            sample_data += f"\n... and {result_count - 5} more rows"
        
        return f"🔍 Found {result_count} rows (out of {total_count}) where {column_name} {condition.replace('_', ' ')} '{value}':\n\n{sample_data}"
        
    except Exception as e:
        return f"❌ Error filtering data: {str(e)}"


# =============================================================================
# 📈 DATA VISUALIZATION TOOLS
# =============================================================================

def setup_plot_style():
    """Set up consistent plot styling."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100


@function_tool
def create_bar_chart(column_name: str, value_column: str = None, save_path: str = None) -> str:
    """
    Create a bar chart showing counts or values by category.
    
    Args:
        column_name (str): Categorical column for x-axis
        value_column (str): Optional numeric column for y-axis (if not provided, shows counts)
        save_path (str): Optional path to save the chart image
        
    Returns:
        str: Success message with chart description or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        setup_plot_style()
        plt.figure(figsize=(12, 6))
        
        if value_column is None:
            # Count plot
            value_counts = current_dataset[column_name].value_counts()
            if len(value_counts) > 20:
                value_counts = value_counts.head(20)
                title_suffix = " (Top 20)"
            else:
                title_suffix = ""
            
            bars = plt.bar(range(len(value_counts)), value_counts.values, color='lightcoral', alpha=0.8)
            plt.title(f'📊 Count of Records by {column_name}{title_suffix}', fontsize=16, fontweight='bold')
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            plt.xticks(range(len(value_counts)), value_counts.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count in zip(bars, value_counts.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
            
            chart_desc = f"bar chart showing counts for each category in '{column_name}'"
            
        else:
            # Value plot
            if value_column not in current_dataset.columns:
                available_cols = ', '.join(current_dataset.columns)
                return f"❌ Value column '{value_column}' not found. Available columns: {available_cols}"
            
            # Group by category and aggregate
            grouped = current_dataset.groupby(column_name)[value_column].mean().sort_values(ascending=False)
            if len(grouped) > 20:
                grouped = grouped.head(20)
                title_suffix = " (Top 20)"
            else:
                title_suffix = ""
            
            bars = plt.bar(range(len(grouped)), grouped.values, color='lightgreen', alpha=0.8)
            plt.title(f'📊 Average {value_column} by {column_name}{title_suffix}', fontsize=16, fontweight='bold')
            plt.xlabel(column_name, fontsize=12)
            plt.ylabel(f'Average {value_column}', fontsize=12)
            plt.xticks(range(len(grouped)), grouped.index, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, grouped.values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(grouped.values), 
                        f'{value:.1f}', ha='center', va='bottom')
            
            chart_desc = f"bar chart showing average {value_column} for each {column_name}"
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"📊 Bar chart saved to '{save_path}'. Shows {chart_desc}"
        else:
            save_path = f"bar_chart_{column_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"📊 Bar chart created and saved as '{save_path}'. Shows {chart_desc}"
        
    except Exception as e:
        return f"❌ Error creating bar chart: {str(e)}"


@function_tool
def create_scatter_plot(x_column: str, y_column: str, color_column: str = None, save_path: str = None) -> str:
    """
    Create a scatter plot to show relationship between two numeric columns.
    
    Args:
        x_column (str): Column for x-axis
        y_column (str): Column for y-axis  
        color_column (str): Optional column to color points by
        save_path (str): Optional path to save the chart image
        
    Returns:
        str: Success message with chart description or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    for col in [x_column, y_column]:
        if col not in current_dataset.columns:
            available_cols = ', '.join(current_dataset.columns)
            return f"❌ Column '{col}' not found. Available columns: {available_cols}"
    
    try:
        # Prepare data
        plot_data = current_dataset[[x_column, y_column]].copy()
        
        # Convert to numeric
        plot_data[x_column] = pd.to_numeric(plot_data[x_column], errors='coerce')
        plot_data[y_column] = pd.to_numeric(plot_data[y_column], errors='coerce')
        
        # Remove rows with missing data
        plot_data = plot_data.dropna()
        
        if len(plot_data) == 0:
            return f"❌ No valid numeric data found for scatter plot between '{x_column}' and '{y_column}'"
        
        setup_plot_style()
        plt.figure(figsize=(10, 8))
        
        if color_column and color_column in current_dataset.columns:
            # Colored scatter plot
            color_data = current_dataset.loc[plot_data.index, color_column]
            scatter = plt.scatter(plot_data[x_column], plot_data[y_column], 
                                c=color_data.astype('category').cat.codes, 
                                alpha=0.7, s=60, cmap='viridis')
            plt.colorbar(scatter, label=color_column)
            title_color = f" (colored by {color_column})"
        else:
            # Simple scatter plot
            plt.scatter(plot_data[x_column], plot_data[y_column], alpha=0.7, s=60, color='steelblue')
            title_color = ""
        
        plt.title(f'📈 {y_column} vs {x_column}{title_color}', fontsize=16, fontweight='bold')
        plt.xlabel(x_column, fontsize=12)
        plt.ylabel(y_column, fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = plot_data[x_column].corr(plot_data[y_column])
        plt.text(0.02, 0.98, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"📈 Scatter plot saved to '{save_path}'. Shows relationship between {x_column} and {y_column} (correlation: {correlation:.3f})"
        else:
            save_path = f"scatter_{x_column}_vs_{y_column}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"📈 Scatter plot created and saved as '{save_path}'. Shows relationship between {x_column} and {y_column} (correlation: {correlation:.3f})"
        
    except Exception as e:
        return f"❌ Error creating scatter plot: {str(e)}"


@function_tool
def create_box_plot(column_name: str, group_column: str = None, save_path: str = None) -> str:
    """
    Create a box plot to show distribution and outliers.
    
    Args:
        column_name (str): Numeric column to analyze
        group_column (str): Optional categorical column to group by
        save_path (str): Optional path to save the chart image
        
    Returns:
        str: Success message with chart description or error message
    """
    if current_dataset is None:
        return "❌ No dataset loaded. Please load a CSV file first."
    
    if column_name not in current_dataset.columns:
        available_cols = ', '.join(current_dataset.columns)
        return f"❌ Column '{column_name}' not found. Available columns: {available_cols}"
    
    try:
        setup_plot_style()
        plt.figure(figsize=(12, 6))
        
        # Convert to numeric
        numeric_data = pd.to_numeric(current_dataset[column_name], errors='coerce')
        
        if group_column and group_column in current_dataset.columns:
            # Grouped box plot
            plot_data = current_dataset[[column_name, group_column]].copy()
            plot_data[column_name] = numeric_data
            plot_data = plot_data.dropna()
            
            if len(plot_data) == 0:
                return f"❌ No valid numeric data for box plot of '{column_name}'"
            
            # Create box plot grouped by category
            groups = plot_data.groupby(group_column)[column_name].apply(list)
            
            plt.boxplot(groups.values, labels=groups.index)
            plt.title(f'📊 Distribution of {column_name} by {group_column}', fontsize=16, fontweight='bold')
            plt.xlabel(group_column, fontsize=12)
            plt.xticks(rotation=45, ha='right')
            chart_desc = f"box plot showing {column_name} distribution across different {group_column} categories"
            
        else:
            # Simple box plot
            clean_data = numeric_data.dropna()
            
            if len(clean_data) == 0:
                return f"❌ No valid numeric data for box plot of '{column_name}'"
            
            plt.boxplot(clean_data)
            plt.title(f'📊 Distribution of {column_name}', fontsize=16, fontweight='bold')
            chart_desc = f"box plot showing {column_name} distribution"
        
        plt.ylabel(column_name, fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"📊 Box plot saved to '{save_path}'. Shows {chart_desc}"
        else:
            save_path = f"box_plot_{column_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return f"📊 Box plot created and saved as '{save_path}'. Shows {chart_desc}"
        
    except Exception as e:
        return f"❌ Error creating box plot: {str(e)}"



# Update the tools list to include visualization tools
AVAILABLE_TOOLS = [
    investigate_directory,
    load_csv_file,
    get_column_names,
    get_dataset_info,
    calculate_column_average,
    calculate_column_stats,
    count_rows_with_value,
    get_unique_values,
    find_max_value,
    find_min_value,
    group_by_column_and_aggregate,
    filter_data,
    create_bar_chart,
    create_scatter_plot,
    create_box_plot
]
