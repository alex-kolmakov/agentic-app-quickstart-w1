"""
Multi-Agent CSV Data Analysis System with Visualization - Main Entry Point

This is the main entry point for the enhanced CSV data analysis system with visualization capabilities.
It provides a specialized multi-agent approach for analyzing CSV data with natural language queries
and creating insightful visualizations.

🎯 Features:
- 📊 Load and analyze CSV files
- 🗣️ Natural language interface
- � Advanced data visualization (charts, plots, dashboards)
- �🔍 Statistical analysis and insights
- 💬 Memory-enabled conversations
- 🛡️ Robust error handling

🏗️ Multi-Agent Architecture:
- 📁 DataLoaderAgent: File operations and data preparation
- 📊 AnalyticsAgent: Statistical calculations and analysis
- 📈 VisualizationAgent: Chart creation and visual insights
- 💬 CommunicationAgent: User-friendly response formatting

🌟 Visualization Capabilities:
- 📊 Histograms for distribution analysis
- 📈 Bar charts for categorical comparisons
- 🔍 Scatter plots for relationship exploration
- 📦 Box plots for outlier detection
- 🌡️ Correlation heatmaps
- 📋 Comprehensive dashboards

Run this file to start the interactive visualization system!
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the package
sys.path.append(str(Path(__file__).parent.parent.parent))

from agentic_app_quickstart.week_1.solution.custom_agents import (
    run_multi_agent_conversation,
    run_demo
)


def print_welcome():
    """Print the welcome message and system overview."""
    print("🎪📈" + "=" * 58 + "🎪📈")
    print("    MULTI-AGENT CSV DATA ANALYSIS & VISUALIZATION SYSTEM")
    print("🎪📈" + "=" * 58 + "🎪📈")
    print()
    print("🚀 Welcome to your intelligent data analysis & visualization assistant!")
    print()
    print("🎯 THE CHALLENGE:")
    print("   Build a system that can load CSV files, understand natural")
    print("   language questions, provide insights, and create visualizations.")
    print()
    print("🏗️ MULTI-AGENT ARCHITECTURE:")
    print("   🤖🤖🤖📈 Specialized agents working together:")
    print("   • 📁 DataLoader - File operations and data preparation")
    print("   • 📊 Analytics - Statistical calculations and analysis") 
    print("   • 📈 Visualization - Chart creation and visual insights")
    print("   • 💬 Communication - User-friendly response formatting")
    print()
    print("✨ FEATURES IMPLEMENTED:")
    print("   ✅ CSV file loading and validation")
    print("   ✅ Natural language question understanding") 
    print("   ✅ Statistical analysis (mean, median, min, max, etc.)")
    print("   ✅ Data filtering and grouping")
    print("   ✅ 📈 Data visualization with multiple chart types")
    print("   ✅ 📋 Comprehensive dashboards")
    print("   ✅ 🎨 Chart export as high-quality images")
    print("   ✅ Error handling and user-friendly messages")
    print("   ✅ Conversation memory (Silver bonus!)")
    print("   ✅ Multi-agent architecture (Bronze bonus!)")
    print("   ✅ Advanced visualization (Gold bonus!)")
    print()
    print("🎨 VISUALIZATION CAPABILITIES:")
    print("   • 📊 Histograms - Show data distributions")
    print("   • 📈 Bar Charts - Compare categories and values")
    print("   • 🔍 Scatter Plots - Explore relationships between variables")
    print("   • 📦 Box Plots - Detect outliers and show distributions")
    print("   • 🌡️ Correlation Heatmaps - Visualize variable relationships")
    print("   • 📋 Summary Dashboards - Comprehensive data overviews")
    print()
    print("🧰 AVAILABLE FUNCTIONS:")
    print("   Data Operations:")
    print("   • load_csv_file() - Load CSV files")
    print("   • get_column_names() - List all columns")
    print("   • get_dataset_info() - Dataset overview")
    print("   • get_unique_values() - Find unique values")
    print()
    print("   Statistical Analysis:")
    print("   • calculate_column_average() - Get mean values")
    print("   • calculate_column_stats() - Comprehensive statistics")
    print("   • find_max_value() / find_min_value() - Find extremes")
    print("   • count_rows_with_value() - Count specific values")
    print("   • group_by_column_and_aggregate() - Group and aggregate")
    print("   • filter_data() - Filter based on conditions")
    print()
    print("   Visualization:")
    print("   • create_histogram() - Distribution charts")
    print("   • create_bar_chart() - Category comparisons")
    print("   • create_scatter_plot() - Relationship exploration")
    print("   • create_box_plot() - Outlier detection")
    print("   • create_correlation_heatmap() - Variable relationships")
    print("   • create_data_summary_dashboard() - Comprehensive overview")
    print()
    print("💡 EXAMPLE QUESTIONS:")
    print('   Data Loading:')
    print('   "Load the employee_data.csv file"')
    print('   "What are the column names?"')
    print()
    print('   Statistical Analysis:')
    print('   "What\'s the average salary?"')
    print('   "How many employees are in Engineering?"')
    print('   "Who has the highest performance score?"')
    print()
    print('   Visualization Requests:')
    print('   "Create a dashboard for the employee data"')
    print('   "Show me a histogram of salaries"')
    print('   "Make a scatter plot of salary vs performance"')
    print('   "Create a bar chart showing average salary by department"')
    print('   "Show a box plot of performance scores by department"')
    print('   "Generate a correlation heatmap"')
    print()
    print("🎪📈" + "=" * 58 + "🎪📈")


async def main():
    """Main function to run the multi-agent data analysis & visualization system."""
    print_welcome()
    
    print("\n🚀 Choose your experience:")
    print("1. 🤖🤖🤖📈 Interactive Multi-Agent System")
    print("2. 🎪 Quick Visualization Demo")
    print("3. ℹ️ About This System")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🤖🤖🤖📈 Starting Multi-Agent Visualization System...")
            await run_multi_agent_conversation()
            break
        elif choice == "2":
            print("\n🎪 Starting Visualization Demo...")
            await run_demo()
            break
        elif choice == "3":
            print_about_system()
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def print_about_system():
    """Print detailed information about the system architecture."""
    print("\n" + "ℹ️" + "=" * 55 + "ℹ️")
    print("              ABOUT THIS SYSTEM")
    print("ℹ️" + "=" * 55 + "ℹ️")
    print()
    print("🏗️ MULTI-AGENT ARCHITECTURE:")
    print()
    print("� DATA LOADER AGENT:")
    print("   • Specializes in CSV file operations")
    print("   • Handles file loading, validation, and error management")
    print("   • Provides dataset metadata and structure information")
    print("   • Acts as the entry point for data operations")
    print()
    print("📊 ANALYTICS AGENT:")
    print("   • Performs statistical calculations and analysis")
    print("   • Calculates means, medians, distributions, and aggregations")
    print("   • Handles data filtering and conditional analysis")
    print("   • Identifies patterns and generates insights")
    print()
    print("📈 VISUALIZATION AGENT:")
    print("   • Creates charts, plots, and visual representations")
    print("   • Supports 6 different visualization types")
    print("   • Exports high-quality images automatically")
    print("   • Chooses appropriate chart types for data")
    print("   • Generates comprehensive dashboards")
    print()
    print("💬 COMMUNICATION AGENT:")
    print("   • Translates technical results into plain English")
    print("   • Provides context and interpretation")
    print("   • Maintains user-friendly conversation flow")
    print("   • Integrates insights from all other agents")
    print()
    print("🛠️ VISUALIZATION TOOLS:")
    print()
    print("📊 HISTOGRAMS:")
    print("   • Show data distributions and frequency patterns")
    print("   • Include statistical summaries (mean, median, std dev)")
    print("   • Automatically handle non-numeric data")
    print()
    print("📈 BAR CHARTS:")
    print("   • Compare categories or show aggregated values")
    print("   • Support both count and value-based comparisons")
    print("   • Include value labels for clarity")
    print()
    print("🔍 SCATTER PLOTS:")
    print("   • Explore relationships between two variables")
    print("   • Show correlation coefficients")
    print("   • Support color-coding by third variable")
    print()
    print("📦 BOX PLOTS:")
    print("   • Detect outliers and show distributions")
    print("   • Compare distributions across groups")
    print("   • Highlight quartiles and median values")
    print()
    print("🌡️ CORRELATION HEATMAPS:")
    print("   • Visualize relationships between all numeric variables")
    print("   • Use color coding for correlation strength")
    print("   • Show correlation coefficients in cells")
    print()
    print("📋 SUMMARY DASHBOARDS:")
    print("   • Comprehensive 4-panel overview")
    print("   • Dataset info, distributions, categories, missing data")
    print("   • Perfect for initial data exploration")
    print()
    print("💾 MEMORY & CONTEXT:")
    print("   • Powered by SQLiteSession for conversation history")
    print("   • Remembers previous questions and datasets")
    print("   • Enables follow-up questions and context references")
    print("   • Maintains separate sessions for different conversations")
    print()
    print("🔧 TECHNICAL IMPLEMENTATION:")
    print("   • Built on OpenAI Agents framework")
    print("   • Uses function_tool decorator for tool integration")
    print("   • Pandas for robust CSV data manipulation")
    print("   • Matplotlib & Seaborn for visualization")
    print("   • Async/await for efficient operation")
    print("   • Comprehensive error handling")
    print()
    print("📁 SAMPLE DATA:")
    print("   • employee_data.csv - Employee information (17 records)")
    print("   • sample_sales.csv - Sales transaction data (21 records)")
    print("   • weather_data.csv - Weather measurements")
    print()
    print("🎯 ASSIGNMENT REQUIREMENTS EXCEEDED:")
    print("   ✅ CSV File Loading")
    print("   ✅ Function Calling (17 tools implemented)")
    print("   ✅ Natural Language Interface")
    print("   ✅ Error Handling")
    print("   🥉 Bronze: Multi-Agent Architecture")
    print("   🥈 Silver: Short-Term Memory")
    print("   🥇 Gold: Advanced Visualization & Analytics")
    print("   🏆 Platinum: Interactive Dashboards & Chart Export")
    print()
    print("ℹ️" + "=" * 55 + "ℹ️")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Thanks for using the Multi-Agent Data Analysis & Visualization System!")
        print("🎨 Don't forget to check your directory for any generated charts!")
    except Exception as e:
        print(f"\n❌ An unexpected error occurred: {str(e)}")
        print("Please check your setup and try again.")
