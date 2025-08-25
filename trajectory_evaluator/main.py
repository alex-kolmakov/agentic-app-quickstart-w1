"""
Trajectory Evaluation for Multi-Agent System

This application evaluates agent trajectories captured by Phoenix to identify:
- Duplicate tool usage
- Agent handoff loops  
- Inefficient routing patterns
- Overall trajectory quality

Based on Arize Phoenix documentation for trace evaluation.
"""

import os
import sys
import asyncio
import argparse
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path

# Phoenix and evaluation imports
import phoenix as px
from phoenix.evals import llm_classify, OpenAIModel
from phoenix.trace import SpanEvaluations
import nest_asyncio

# Local imports
from trajectory_analyzer import TrajectoryAnalyzer
from evaluation_prompts import TRAJECTORY_PROMPTS
from utils import setup_logging, export_results, print_evaluation_summary

# Allow nested event loops in notebooks/async environments
nest_asyncio.apply()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate agent trajectories for efficiency and correctness")
    parser.add_argument("--lookback-days", type=int, default=1, 
                       help="Number of days to look back for traces (default: 1)")
    parser.add_argument("--phoenix-endpoint", type=str, default=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"),
                       help="Phoenix endpoint URL")
    parser.add_argument("--project-name", type=str, default=os.getenv("PHOENIX_PROJECT_NAME", "agentic-app-quickstart"),
                       help="Phoenix project name to analyze")
    parser.add_argument("--output-dir", type=str, default="./outputs",
                       help="Directory to save evaluation results")
    parser.add_argument("--evaluation-types", nargs="+", 
                       choices=["duplicate_tools", "handoff_loops", "efficiency", "overall"],
                       default=["duplicate_tools", "handoff_loops", "efficiency"],
                       help="Types of evaluations to run")
    parser.add_argument("--dry-run", action="store_true",
                       help="Run analysis without uploading results to Phoenix")
    
    return parser.parse_args()


class TrajectoryEvaluator:
    """Main class for evaluating agent trajectories."""
    
    def __init__(self, phoenix_endpoint: str, project_name: str, output_dir: str):
        self.phoenix_endpoint = phoenix_endpoint
        self.project_name = project_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup Phoenix client
        self.phoenix_client = px.Client(endpoint=phoenix_endpoint)
        
        # Setup OpenAI model for evaluations
        self.openai_model = OpenAIModel(
            api_key=os.environ["OPENAI_API_KEY"],
            model="gpt-4o-mini",
            temperature=0.0,
            base_url=os.getenv("OPENAI_API_ENDPOINT", "https://api.openai.com/v1")
        )
        
        # Initialize trajectory analyzer
        self.analyzer = TrajectoryAnalyzer()
        
        # Setup logging
        setup_logging(self.output_dir / "evaluation.log")
        
    def fetch_trace_data(self, lookback_days: int) -> pd.DataFrame:
        """Fetch trace data from Phoenix for the specified time period."""
        print(f"🔍 Fetching traces from Phoenix project '{self.project_name}'...")
        print(f"   📅 Looking back {lookback_days} days")
        
        try:
            # Get spans dataframe from Phoenix
            spans_df = self.phoenix_client.get_spans_dataframe(
                project_name=self.project_name,
                start_time=datetime.now(timezone.utc) - timedelta(days=lookback_days),
                end_time=datetime.now(timezone.utc)
            )
            
            print(f"   ✅ Fetched {len(spans_df)} spans")
            return spans_df
            
        except Exception as e:
            print(f"   ❌ Failed to fetch trace data: {e}")
            return pd.DataFrame()
    
    def filter_agent_spans(self, spans_df: pd.DataFrame) -> pd.DataFrame:
        """Filter spans to get only agent-related spans with tools and handoffs."""
        print("🔍 Filtering agent-related spans...")
        
        # Filter for spans that represent agent actions (tool calls, handoffs)
        # Handle NaN values by filling them with empty strings first
        span_kind_col = spans_df['attributes.openinference.span.kind'].fillna('')
        
        agent_spans = spans_df[
            (spans_df['name'].str.contains('Agent|Tool|Function', case=False, na=False)) |
            (span_kind_col.isin(['AGENT', 'TOOL', 'CHAIN']))
        ].copy()
        
        print(f"   ✅ Found {len(agent_spans)} agent-related spans")
        return agent_spans
    
    def prepare_evaluation_data(self, spans_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare spans data for trajectory evaluation."""
        print("📊 Preparing data for trajectory evaluation...")
        
        # Group spans by trace to reconstruct trajectories
        trajectories = []
        
        for trace_id, trace_spans in spans_df.groupby('context.trace_id'):
            # Sort spans by start time to get chronological order
            trace_spans = trace_spans.sort_values('start_time')
            
            # Extract trajectory information
            trajectory_info = self.analyzer.extract_trajectory(trace_spans)
            
            if trajectory_info:
                trajectories.append({
                    'trace_id': trace_id,
                    'root_span_id': trace_spans[trace_spans['parent_id'].isna()]['context.span_id'].iloc[0] if any(trace_spans['parent_id'].isna()) else trace_spans['context.span_id'].iloc[0],
                    **trajectory_info
                })
        
        eval_df = pd.DataFrame(trajectories)
        print(f"   ✅ Prepared {len(eval_df)} trajectories for evaluation")
        return eval_df
    
    def run_evaluation(self, eval_df: pd.DataFrame, evaluation_type: str) -> pd.DataFrame:
        """Run a specific type of trajectory evaluation."""
        print(f"🧠 Running {evaluation_type} evaluation...")
        
        if evaluation_type not in TRAJECTORY_PROMPTS:
            raise ValueError(f"Unknown evaluation type: {evaluation_type}")
        
        template = TRAJECTORY_PROMPTS[evaluation_type]["template"]
        rails = TRAJECTORY_PROMPTS[evaluation_type]["rails"]
        
        try:
            results = llm_classify(
                dataframe=eval_df,
                template=template,
                model=self.openai_model,
                rails=rails,
                provide_explanation=True
            )
            
            print(f"   ✅ Completed {evaluation_type} evaluation for {len(results)} trajectories")
            return results
            
        except Exception as e:
            print(f"   ❌ Failed {evaluation_type} evaluation: {e}")
            return pd.DataFrame()
    
    def analyze_results(self, eval_df: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze evaluation results and generate insights."""
        print("📈 Analyzing evaluation results...")
        
        analysis = {
            'total_trajectories': len(eval_df),
            'evaluation_summary': {},
            'insights': [],
            'recommendations': []
        }
        
        for eval_type, eval_results in results.items():
            if eval_results.empty:
                continue
                
            # Merge with original data
            merged = eval_df.merge(eval_results, left_index=True, right_index=True, how='left')
            
            # Calculate metrics
            total = len(merged)
            if 'label' in eval_results.columns:
                problem_count = len(merged[merged['label'].isin(['incorrect', 'inefficient', 'duplicate', 'loop'])])
                problem_rate = problem_count / total if total > 0 else 0
                
                analysis['evaluation_summary'][eval_type] = {
                    'total_evaluated': total,
                    'problems_found': problem_count,
                    'problem_rate': problem_rate,
                    'quality_score': 1.0 - problem_rate
                }
                
                # Generate insights
                if problem_rate > 0.2:  # More than 20% problematic
                    analysis['insights'].append(f"High {eval_type} problem rate: {problem_rate:.1%}")
                    
                # Specific recommendations based on evaluation type
                if eval_type == 'duplicate_tools' and problem_rate > 0.1:
                    analysis['recommendations'].append("Review tool usage patterns to reduce duplicates")
                elif eval_type == 'handoff_loops' and problem_rate > 0.05:
                    analysis['recommendations'].append("Optimize agent handoff logic to prevent loops")
                elif eval_type == 'efficiency' and problem_rate > 0.3:
                    analysis['recommendations'].append("Streamline agent trajectories for better efficiency")
        
        return analysis
    
    def save_results(self, eval_df: pd.DataFrame, results: Dict[str, pd.DataFrame], 
                    analysis: Dict[str, Any]) -> None:
        """Save evaluation results to files."""
        print("💾 Saving evaluation results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw trajectory data
        eval_df.to_csv(self.output_dir / f"trajectories_{timestamp}.csv", index=False)
        
        # Save evaluation results
        for eval_type, eval_results in results.items():
            if not eval_results.empty:
                eval_results.to_csv(self.output_dir / f"{eval_type}_results_{timestamp}.csv")
        
        # Save analysis summary
        with open(self.output_dir / f"analysis_summary_{timestamp}.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"   ✅ Results saved to {self.output_dir}")
    
    def upload_to_phoenix(self, eval_df: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> None:
        """Upload evaluation results back to Phoenix."""
        print("📤 Uploading evaluation results to Phoenix...")
        
        try:
            for eval_type, eval_results in results.items():
                if eval_results.empty:
                    continue
                
                # Prepare data for Phoenix
                phoenix_eval_df = eval_results.copy()
                phoenix_eval_df.index = eval_df['root_span_id']
                phoenix_eval_df.index.name = 'context.span_id'
                
                # Rename columns for Phoenix
                if 'label' in phoenix_eval_df.columns:
                    phoenix_eval_df = phoenix_eval_df.rename(columns={
                        'label': 'label',
                        'explanation': 'explanation'
                    })
                
                # Upload to Phoenix
                span_evaluations = SpanEvaluations(
                    eval_name=f"trajectory_{eval_type}",
                    dataframe=phoenix_eval_df
                )
                
                self.phoenix_client.log_evaluations(span_evaluations)
                print(f"   ✅ Uploaded {eval_type} evaluations to Phoenix")
                
        except Exception as e:
            print(f"   ❌ Failed to upload to Phoenix: {e}")


async def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("🚀 Starting Trajectory Evaluation")
    print(f"   📡 Phoenix: {args.phoenix_endpoint}")
    print(f"   📁 Project: {args.project_name}")
    print(f"   📅 Lookback: {args.lookback_days} days")
    print(f"   🧪 Evaluations: {', '.join(args.evaluation_types)}")
    print()
    
    # Validate environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return 1
    
    try:
        # Initialize evaluator
        evaluator = TrajectoryEvaluator(
            phoenix_endpoint=args.phoenix_endpoint,
            project_name=args.project_name,
            output_dir=args.output_dir
        )
        
        # Fetch trace data
        spans_df = evaluator.fetch_trace_data(args.lookback_days)
        if spans_df.empty:
            print("❌ No trace data found. Make sure your application is generating traces.")
            return 1
        
        # Filter for agent spans
        agent_spans = evaluator.filter_agent_spans(spans_df)
        if agent_spans.empty:
            print("❌ No agent-related spans found.")
            return 1
        
        # Prepare evaluation data
        eval_df = evaluator.prepare_evaluation_data(agent_spans)
        if eval_df.empty:
            print("❌ No trajectories could be extracted.")
            return 1
        
        # Run evaluations
        results = {}
        for eval_type in args.evaluation_types:
            result = evaluator.run_evaluation(eval_df, eval_type)
            if not result.empty:
                results[eval_type] = result
        
        if not results:
            print("❌ No evaluation results generated.")
            return 1
        
        # Analyze results
        analysis = evaluator.analyze_results(eval_df, results)
        
        # Save results
        evaluator.save_results(eval_df, results, analysis)
        
        # Upload to Phoenix (unless dry run)
        if not args.dry_run:
            evaluator.upload_to_phoenix(eval_df, results)
        else:
            print("🏃 Dry run mode - skipping Phoenix upload")
        
        # Print summary
        print_evaluation_summary(analysis)
        
        print("\n✅ Trajectory evaluation completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
