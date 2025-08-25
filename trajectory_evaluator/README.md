# Trajectory Evaluator for Multi-Agent System

Evaluates agent trajectories from Phoenix traces to identify duplicate tool usage, handoff loops, and inefficient patterns.

## Quick Start

### Prerequisites
- OpenAI API key in project root `.env` file
- Phoenix running (via docker-compose)
- Multi-agent app generating traces

### Run Evaluation

```bash
# Evaluate last day's traces
python main.py

# Dry run (no upload to Phoenix)
python main.py --dry-run

# Custom options
python main.py --lookback-days 7 --evaluation-types duplicate_tools handoff_loops
```

### Docker Usage

```bash
# Run with docker-compose
docker-compose up trajectory-evaluator

# Or run manually
docker-compose run trajectory-evaluator python main.py --lookback-days 1
```

## Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --lookback-days INT          Days to look back for traces (default: 1)
  --project-name TEXT          Phoenix project name (default: agentic-app-quickstart)
  --phoenix-endpoint TEXT      Phoenix endpoint URL (default: http://localhost:6006)
  --output-dir TEXT           Directory for results (default: ./outputs)
  --evaluation-types LIST     Types to run: duplicate_tools, handoff_loops, efficiency, overall
  --dry-run                   Run analysis without uploading to Phoenix
```

## Files

- `main.py` - Main evaluation orchestrator
- `trajectory_analyzer.py` - Core analysis logic
- `evaluation_prompts.py` - LLM evaluation prompts
- `config.py` - Configuration management
- `utils.py` - Utility functions
- `outputs/` - Results directory

## Output

Results saved to `outputs/` with timestamps:
- `trajectories_*.csv` - Raw trajectory data
- `*_results_*.csv` - Evaluation results
- `analysis_summary_*.json` - Combined analysis
- `evaluation_report_*.md` - Human-readable report

View results in Phoenix UI at http://localhost:6006
