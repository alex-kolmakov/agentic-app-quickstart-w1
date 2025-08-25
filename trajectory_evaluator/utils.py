"""
Utility functions for trajectory evaluation.
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def setup_logging(log_file: Path) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def export_results(results: Dict[str, Any], output_path: Path) -> None:
    """Export evaluation results to various formats."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export as JSON
    json_path = output_path / f"trajectory_evaluation_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Export summary as CSV if applicable
    if 'evaluation_summary' in results:
        summary_df = pd.DataFrame(results['evaluation_summary']).T
        csv_path = output_path / f"evaluation_summary_{timestamp}.csv"
        summary_df.to_csv(csv_path)
    
    print(f"Results exported to {output_path}")


def format_trajectory_for_display(trajectory: Dict[str, Any]) -> str:
    """Format trajectory data for human-readable display."""
    lines = []
    lines.append(f"📝 User Request: {trajectory.get('user_input', 'N/A')}")
    lines.append(f"🤖 Agents: {' → '.join(trajectory.get('agents_sequence', []))}")
    lines.append(f"🔧 Tools: {', '.join(trajectory.get('tool_calls', []))}")
    lines.append(f"📊 Metrics:")
    lines.append(f"   • Tool calls: {trajectory.get('total_tool_calls', 0)}")
    lines.append(f"   • Unique tools: {trajectory.get('unique_tools', 0)}")
    lines.append(f"   • Handoffs: {trajectory.get('agent_handoffs', 0)}")
    lines.append(f"   • Duplicates: {trajectory.get('duplicate_tools', 0)}")
    
    return "\n".join(lines)


def validate_environment() -> bool:
    """Validate that required environment variables are set."""
    import os
    
    required_vars = [
        "OPENAI_API_KEY"
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {', '.join(missing)}")
        return False
    
    return True


def create_evaluation_report(analysis: Dict[str, Any], output_path: Path) -> None:
    """Create a comprehensive evaluation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Trajectory Evaluation Report
Generated: {timestamp}

## Summary
Total Trajectories Evaluated: {analysis.get('total_trajectories', 0)}

## Evaluation Results
"""
    
    for eval_type, summary in analysis.get('evaluation_summary', {}).items():
        quality_score = summary.get('quality_score', 0)
        problem_rate = summary.get('problem_rate', 0)
        
        report += f"""
### {eval_type.replace('_', ' ').title()}
- Quality Score: {quality_score:.1%}
- Problem Rate: {problem_rate:.1%}
- Issues Found: {summary.get('problems_found', 0)}/{summary.get('total_evaluated', 0)}
"""
    
    if analysis.get('insights'):
        report += "\n## Key Insights\n"
        for insight in analysis['insights']:
            report += f"- {insight}\n"
    
    if analysis.get('recommendations'):
        report += "\n## Recommendations\n"
        for rec in analysis['recommendations']:
            report += f"- {rec}\n"
    
    # Save report
    report_path = output_path / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"📋 Evaluation report saved to {report_path}")


def print_evaluation_summary(analysis: Dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "="*60)
    print("📊 TRAJECTORY EVALUATION SUMMARY")
    print("="*60)
    
    total = analysis.get('total_trajectories', 0)
    print(f"\n📈 Total Trajectories Analyzed: {total}")
    
    if 'evaluation_summary' in analysis:
        print("\n🎯 Quality Scores:")
        for eval_type, summary in analysis['evaluation_summary'].items():
            score = summary.get('quality_score', 0)
            problems = summary.get('problems_found', 0)
            emoji = "✅" if score > 0.8 else "⚠️" if score > 0.6 else "❌"
            print(f"   {emoji} {eval_type.replace('_', ' ').title()}: {score:.1%} ({problems} issues)")
    
    if analysis.get('insights'):
        print("\n💡 Key Insights:")
        for insight in analysis['insights']:
            print(f"   • {insight}")
    
    if analysis.get('recommendations'):
        print("\n🎯 Recommendations:")
        for rec in analysis['recommendations']:
            print(f"   • {rec}")
    
    print("\n" + "="*60)
