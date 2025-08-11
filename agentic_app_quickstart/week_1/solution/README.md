# 🎪📈 Multi-Agent CSV Data Analysis & Visualization System

A comprehensive solution for the Week 1 Assignment that creates intelligent systems capable of loading CSV files, understanding natural language questions, providing clear insights, and generating beautiful visualizations about data.

## 🏆 Features Implemented

### ✅ Core Requirements
- **CSV File Loading**: Robust file loading with error handling and validation
- **Function Calling**: 17 specialized tools for comprehensive data analysis and visualization
- **Natural Language Interface**: Understands and responds to human questions
- **Error Handling**: Graceful error handling with helpful user messages

### 🌟 Bonus Features
- **🥉 Bronze - Multi-Agent Architecture**: 4 specialized agents working together
- **🥈 Silver - Short-Term Memory**: Conversation memory using SQLiteSession
- **🥇 Gold - Advanced Visualization**: 6 chart types with professional styling
- **🏆 Platinum - Interactive Dashboards**: Comprehensive visual summaries

## 🏗️ Multi-Agent Architecture

Our system uses **4 specialized agents** working together:

### 📁 DataLoaderAgent
- **Specializes in**: File operations and data preparation
- **Tools**: `load_csv_file`, `get_column_names`, `get_dataset_info`, `get_unique_values`
- **Role**: Entry point for data operations, file validation, metadata inspection

### 📊 AnalyticsAgent
- **Specializes in**: Statistical calculations and data analysis
- **Tools**: `calculate_column_average`, `calculate_column_stats`, `count_rows_with_value`, `find_max_value`, `find_min_value`, `group_by_column_and_aggregate`, `filter_data`
- **Role**: Number crunching, pattern identification, statistical insights

### 📈 VisualizationAgent
- **Specializes in**: Chart creation and visual insights
- **Tools**: `create_histogram`, `create_bar_chart`, `create_scatter_plot`, `create_box_plot`, `create_correlation_heatmap`, `create_data_summary_dashboard`
- **Role**: Visual representation of data, chart export, visual pattern identification

### 💬 CommunicationAgent
- **Specializes in**: User-friendly response formatting
- **Tools**: None (focuses on communication)
- **Role**: Translates technical results into plain English, provides context and insights

## 📈 Visualization Capabilities

### 📊 Histogram
- **Purpose**: Show data distributions and frequency patterns
- **Features**: Statistical summaries, automatic binning, professional styling
- **Use case**: `"Create a histogram of salaries"`

### 📈 Bar Chart
- **Purpose**: Compare categories or show aggregated values
- **Features**: Support for both counts and values, automatic labeling
- **Use case**: `"Make a bar chart showing average salary by department"`

### 🔍 Scatter Plot
- **Purpose**: Explore relationships between two variables
- **Features**: Correlation coefficients, optional color coding
- **Use case**: `"Create a scatter plot of salary vs performance score"`

### 📦 Box Plot
- **Purpose**: Detect outliers and show distributions
- **Features**: Quartile visualization, group comparisons
- **Use case**: `"Show a box plot of performance scores by department"`

### 🌡️ Correlation Heatmap
- **Purpose**: Visualize relationships between all numeric variables
- **Features**: Color-coded correlation matrix, coefficient display
- **Use case**: `"Generate a correlation heatmap"`

### 📋 Summary Dashboard
- **Purpose**: Comprehensive 4-panel data overview
- **Features**: Dataset info, distributions, categories, missing data analysis
- **Use case**: `"Create a comprehensive dashboard"`

## 🛠️ Complete Tool List

### File Operations
- `load_csv_file(file_path)` - Load CSV files with validation
- `get_column_names()` - List all column names
- `get_dataset_info()` - Get comprehensive dataset overview
- `get_unique_values(column_name)` - Find unique values in a column

### Statistical Analysis
- `calculate_column_average(column_name)` - Calculate mean values
- `calculate_column_stats(column_name)` - Full statistical summary
- `find_max_value(column_name)` - Find maximum with context
- `find_min_value(column_name)` - Find minimum with context
- `count_rows_with_value(column_name, value)` - Count specific values
- `group_by_column_and_aggregate(group_col, agg_col, operation)` - Group and aggregate
- `filter_data(column_name, condition, value)` - Filter data with conditions

### Data Visualization
- `create_histogram(column_name, bins, save_path)` - Distribution charts
- `create_bar_chart(column_name, value_column, save_path)` - Category comparisons
- `create_scatter_plot(x_column, y_column, color_column, save_path)` - Relationship exploration
- `create_box_plot(column_name, group_column, save_path)` - Outlier detection
- `create_correlation_heatmap(save_path)` - Variable relationships
- `create_data_summary_dashboard(save_path)` - Comprehensive overview

## 🚀 Quick Start

### 1. Run the System
```bash
python main.py
```

### 2. Choose Your Experience
1. **Interactive Multi-Agent System** - Full conversation interface
2. **Quick Visualization Demo** - Automated demo with sample data

### 3. Example Commands

#### Data Loading
```
"Load the employee_data.csv file"
"What are the column names?"
"Show me a dataset overview"
```

#### Statistical Analysis
```
"What's the average salary?"
"How many employees are in Engineering?"
"Who has the highest performance score?"
"Group employees by department and calculate average salary"
```

#### Visualization Requests
```
"Create a comprehensive dashboard"
"Show me a histogram of salaries"
"Make a scatter plot of salary vs performance score"
"Create a bar chart showing average salary by department"
"Show a box plot of performance scores by department"
"Generate a correlation heatmap"
```

## 📁 Sample Data

The system comes with three sample datasets:

### employee_data.csv
Employee information including:
- name, department, salary, hire_date, performance_score
- 17 employees across Engineering, Marketing, Sales, HR
- Perfect for demonstrating all visualization types

### sample_sales.csv  
Sales transaction data including:
- date, product, price, quantity, customer_state
- 21 transactions across various products and states
- Great for time series and geographical analysis

### weather_data.csv
Weather measurement data for analysis practice

## 💡 Example Multi-Agent Conversation

```
You: Load employee_data.csv and create a dashboard
📁 DataLoaderAgent: ✅ Successfully loaded 'employee_data.csv' with 17 rows and 5 columns.
                    Columns: name, department, salary, hire_date, performance_score

📈 VisualizationAgent: 📊 Data summary dashboard created and saved as 'dashboard_employee_data_20250811_143022.png'.
                       Comprehensive overview of employee_data.csv

💬 CommunicationAgent: Perfect! I've created a comprehensive dashboard that gives you a complete overview 
                       of your employee data. The dashboard shows:
                       
                       📊 Dataset Overview: 17 employees across 4 departments
                       📈 Salary Distribution: Range from $58K to $92K with most around $70K
                       📝 Department Breakdown: Engineering (6), Marketing (2), Sales (2), HR (2)
                       ✅ Data Quality: No missing values - clean dataset!
                       
                       Would you like me to create any specific charts to dive deeper into the data?

You: Show me a scatter plot of salary vs performance
� VisualizationAgent: 📈 Scatter plot created and saved as 'scatter_salary_vs_performance_score_20250811_143045.png'.
                       Shows relationship between salary and performance_score (correlation: 0.756)

� CommunicationAgent: Excellent! The scatter plot reveals a strong positive correlation (0.756) between
                       salary and performance score. This means:
                       
                       🔍 Key insights:
                       • Higher performing employees tend to earn more
                       • The correlation is quite strong, suggesting fair compensation
                       • There's a clear upward trend in the data
                       
                       This suggests your organization rewards performance well! 
                       Would you like to explore this relationship by department?
```

## 🔧 Technical Implementation

### Dependencies
- `openai-agents` - Agent framework for multi-agent coordination
- `pandas` - Data manipulation and analysis
- `matplotlib` - Core plotting functionality
- `seaborn` - Enhanced statistical visualizations
- `numpy` - Numerical computations (via matplotlib)

### Memory System
- Uses `SQLiteSession` for conversation history
- Maintains context across agent handoffs
- Enables follow-up questions and complex analysis workflows

### Visualization Features
- **Professional styling** with consistent color schemes
- **High-quality exports** (300 DPI PNG images)
- **Automatic error handling** for invalid data
- **Smart defaults** for chart parameters
- **Context annotations** with statistics and insights

### Error Handling
- File not found errors with helpful suggestions
- Invalid column name handling with available options
- Non-numeric data handling for statistical/visual operations
- Graceful degradation with user-friendly messages

## 🎯 Learning Outcomes

By implementing this system, you learn:

- **Multi-Agent Architecture**: Specialized agents with clear responsibilities
- **Function Calling**: Extending agent capabilities with custom tools
- **Memory Management**: Maintaining conversation context and state
- **Error Handling**: Building robust, user-friendly systems
- **Data Visualization**: Creating professional charts and dashboards
- **System Design**: Agent coordination and handoff patterns
- **Natural Language Processing**: Understanding user intent for data tasks

## 🚀 Next Steps

Potential enhancements:
- **Interactive visualizations** with Plotly or Bokeh
- **Real-time data streaming** analysis
- **Machine learning insights** and predictions
- **Web interface** with Flask/FastAPI
- **Multiple file format support** (Excel, JSON, Parquet)
- **Data export functionality** to various formats
- **Custom visualization themes** and branding
- **Performance optimization** for large datasets

---

**Built with ❤️ and 📈 for the Agentic App Quickstart Week 1 Assignment**

*Featuring advanced multi-agent architecture with comprehensive data visualization capabilities*