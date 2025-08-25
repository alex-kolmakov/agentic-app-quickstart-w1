"""
Evaluation Prompts for Trajectory Analysis

This module contains LLM evaluation prompts for different aspects of agent trajectory analysis.
Based on Phoenix LLM evaluation patterns.
"""

# Duplicate Tool Usage Evaluation
DUPLICATE_TOOLS_PROMPT = """You are evaluating an AI agent system's trajectory for duplicate tool usage.

You will be given:
1. The sequence of tool calls made by the agents
2. Information about which tools were called multiple times
3. The original user request

Analyze whether the duplicate tool usage is justified or problematic:

ACCEPTABLE duplicates:
- Different parameters for the same tool (e.g., analyzing different columns)
- Tool called in different contexts by different agents
- Iterative refinement based on user feedback

PROBLEMATIC duplicates:
- Identical tool calls with same parameters in sequence
- Tool called multiple times without clear purpose
- Redundant data loading or analysis

User Request: {user_input}

Tool Calls Sequence: {tool_calls}

Agent Sequence: {agents_sequence}

Duplicate Analysis: {duplicate_tools} duplicates found
Tool Usage Details: {max_tool_usage} max usage of {most_used_tool}

Respond with exactly one word: 'acceptable' or 'problematic'.
- 'acceptable' → duplicates are justified and serve different purposes
- 'problematic' → duplicates indicate inefficiency or errors"""

# Agent Handoff Loops Evaluation  
HANDOFF_LOOPS_PROMPT = """You are evaluating an AI agent system's trajectory for handoff loops.

You will be given:
1. The sequence of agents that handled the request
2. Information about agent transitions and potential loops
3. The original user request

Analyze whether the agent handoffs are efficient or contain problematic loops:

EFFICIENT handoffs:
- Clear progression from data loading → analysis → visualization → communication
- Each agent adds value before handing off
- Logical specialization-based routing

PROBLEMATIC loops:
- Agent A → Agent B → Agent A without clear reason
- Multiple unnecessary handoffs for simple requests
- Circular routing without progress

User Request: {user_input}

Agent Sequence: {agents_sequence}

Handoff Count: {agent_handoffs}

Agent Revisits: {agent_revisits}
Immediate Loops: {immediate_agent_loops}

Respond with exactly one word: 'efficient' or 'loop'.
- 'efficient' → handoffs follow logical progression
- 'loop' → contains unnecessary circular routing"""

# Overall Trajectory Efficiency Evaluation
EFFICIENCY_PROMPT = """You are evaluating the overall efficiency of an AI agent system's trajectory.

You will be given:
1. The complete trajectory including agents and tools
2. Various metrics about the execution
3. The original user request

Evaluate whether the trajectory is efficient for accomplishing the user's goal:

EFFICIENT trajectory:
- Minimal unnecessary steps
- Appropriate tool selection for the task
- Logical progression toward the goal
- Reasonable number of operations for task complexity

INEFFICIENT trajectory:
- Too many redundant operations
- Overly complex routing for simple requests
- Tools used inappropriately
- Excessive handoffs or processing steps

User Request: {user_input}

Agent Sequence: {agents_sequence}
Tool Calls: {tool_calls}

Total Tools Used: {total_tool_calls}
Unique Tools: {unique_tools}
Agent Handoffs: {agent_handoffs}
Duplicate Tools: {duplicate_tools}

Respond with exactly one word: 'efficient' or 'inefficient'.
- 'efficient' → trajectory accomplishes the goal with reasonable steps
- 'inefficient' → trajectory has unnecessary complexity or redundancy"""

# Overall Trajectory Quality Evaluation
OVERALL_QUALITY_PROMPT = """You are evaluating the overall quality of an AI agent system's trajectory.

Consider all aspects: correctness, efficiency, and user experience.

You will be given the complete trajectory information and should assess:
1. Does it appear to accomplish the user's goal?
2. Is the approach reasonable and logical?
3. Are the tools and agents used appropriately?
4. Is the overall experience likely to be satisfactory?

User Request: {user_input}

Agent Sequence: {agents_sequence}
Tool Calls: {tool_calls}

Metrics:
- Total Tool Calls: {total_tool_calls}
- Unique Tools: {unique_tools}  
- Agent Handoffs: {agent_handoffs}
- Duplicate Tools: {duplicate_tools}

Respond with exactly one word: 'correct' or 'incorrect'.
- 'correct' → trajectory appears to successfully accomplish the user's goal
- 'incorrect' → trajectory is problematic and unlikely to satisfy the user"""

# Main prompts dictionary
TRAJECTORY_PROMPTS = {
    "duplicate_tools": {
        "template": DUPLICATE_TOOLS_PROMPT,
        "rails": ["acceptable", "problematic"]
    },
    "handoff_loops": {
        "template": HANDOFF_LOOPS_PROMPT,
        "rails": ["efficient", "loop"]
    },
    "efficiency": {
        "template": EFFICIENCY_PROMPT,
        "rails": ["efficient", "inefficient"]
    },
    "overall": {
        "template": OVERALL_QUALITY_PROMPT,
        "rails": ["correct", "incorrect"]
    }
}
