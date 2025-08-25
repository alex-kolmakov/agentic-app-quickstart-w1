"""
Trajectory Analysis Utilities

This module provides utilities for extracting and analyzing agent trajectories
from Phoenix trace data.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
import json
import re
from collections import defaultdict, Counter


class TrajectoryAnalyzer:
    """Analyzes agent trajectories to extract patterns and identify issues."""
    
    def __init__(self):
        self.agent_patterns = {
            'DataLoaderAgent': r'(load|investigate|get_column|get_dataset)',
            'AnalyticsAgent': r'(calculate|stats|aggregate|filter|find_)',
            'VisualizationAgent': r'(create_.*chart|create_.*plot)',
            'CommunicationAgent': r'(format|present|communicate)'
        }
    
    def extract_trajectory(self, trace_spans: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Extract trajectory information from a trace's spans."""
        if trace_spans.empty:
            return None
        
        try:
            # Extract agent sequence
            agents_sequence = self.extract_agent_sequence(trace_spans)
            
            # Extract tool calls
            tool_calls = self.extract_tool_calls(trace_spans)
            
            # Extract user input (from root span)
            user_input = self.extract_user_input(trace_spans)
            
            # Calculate trajectory metrics
            metrics = self.calculate_trajectory_metrics(agents_sequence, tool_calls)
            
            return {
                'user_input': user_input,
                'agents_sequence': agents_sequence,
                'tool_calls': tool_calls,
                'agent_handoffs': len(agents_sequence) - 1 if len(agents_sequence) > 1 else 0,
                'total_tool_calls': len(tool_calls),
                'unique_tools': len(set(tool_calls)),
                'duplicate_tools': len(tool_calls) - len(set(tool_calls)),
                **metrics
            }
            
        except Exception as e:
            print(f"Warning: Failed to extract trajectory: {e}")
            return None
    
    def extract_agent_sequence(self, trace_spans: pd.DataFrame) -> List[str]:
        """Extract the sequence of agents involved in the trace."""
        agents = []
        
        # Look for agent names in span names or attributes
        for _, span in trace_spans.iterrows():
            agent_name = self.identify_agent(span)
            if agent_name and (not agents or agents[-1] != agent_name):
                agents.append(agent_name)
        
        return agents
    
    def identify_agent(self, span: pd.Series) -> Optional[str]:
        """Identify which agent a span belongs to."""
        span_name = str(span.get('name', ''))
        
        # Direct agent name matching
        for agent_name in self.agent_patterns.keys():
            if agent_name.lower() in span_name.lower():
                return agent_name
        
        # Pattern-based matching for tool calls
        for agent_name, pattern in self.agent_patterns.items():
            if re.search(pattern, span_name, re.IGNORECASE):
                return agent_name
        
        # Check attributes for agent information
        attributes = span.get('attributes', {})
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if 'agent' in key.lower():
                    return str(value)
        
        return None
    
    def extract_tool_calls(self, trace_spans: pd.DataFrame) -> List[str]:
        """Extract the sequence of tool calls from the trace."""
        tool_calls = []
        
        for _, span in trace_spans.iterrows():
            tool_name = self.identify_tool_call(span)
            if tool_name:
                tool_calls.append(tool_name)
        
        return tool_calls
    
    def identify_tool_call(self, span: pd.Series) -> Optional[str]:
        """Identify if a span represents a tool call and extract the tool name."""
        span_name = str(span.get('name', ''))
        
        # Common tool call patterns
        tool_patterns = [
            r'tool\.(\w+)',
            r'function\.(\w+)',
            r'(\w+_\w+)',  # function_name pattern
            r'Tool\.(\w+)',
            r'Function\.(\w+)'
        ]
        
        for pattern in tool_patterns:
            match = re.search(pattern, span_name, re.IGNORECASE)
            if match:
                return match.group(1) if match.groups() else span_name
        
        # Check if span kind indicates a tool call
        span_kind = span.get('attributes.openinference.span.kind', '')
        if span_kind in ['TOOL', 'FUNCTION']:
            return span_name
        
        # Look for function names in known tool patterns
        known_tools = [
            'load_csv_file', 'investigate_directory', 'get_column_names',
            'calculate_column_average', 'calculate_column_stats',
            'create_bar_chart', 'create_scatter_plot', 'create_box_plot',
            'filter_data', 'group_by_column_and_aggregate'
        ]
        
        for tool in known_tools:
            if tool in span_name.lower():
                return tool
        
        return None
    
    def extract_user_input(self, trace_spans: pd.DataFrame) -> str:
        """Extract the original user input from the trace."""
        # Look for root span or spans with user input
        root_spans = trace_spans[trace_spans['parent_id'].isna()]
        
        if not root_spans.empty:
            root_span = root_spans.iloc[0]
            
            # Check various attribute locations for user input
            input_fields = [
                'attributes.input.value',
                'attributes.llm.input_messages',
                'attributes.user.message',
                'input.value',
                'user_input'
            ]
            
            for field in input_fields:
                value = root_span.get(field)
                if value and str(value).strip():
                    return str(value)
        
        # Fallback: look for any span with user input
        for _, span in trace_spans.iterrows():
            for field in ['input.value', 'user_input', 'message']:
                value = span.get(field)
                if value and str(value).strip() and len(str(value)) > 10:
                    return str(value)
        
        return "No user input found"
    
    def calculate_trajectory_metrics(self, agents_sequence: List[str], 
                                   tool_calls: List[str]) -> Dict[str, Any]:
        """Calculate various metrics about the trajectory."""
        metrics = {}
        
        # Agent handoff analysis
        if len(agents_sequence) > 1:
            # Check for immediate loops (A -> B -> A)
            immediate_loops = 0
            for i in range(len(agents_sequence) - 2):
                if agents_sequence[i] == agents_sequence[i + 2]:
                    immediate_loops += 1
            
            metrics['immediate_agent_loops'] = immediate_loops
            
            # Check for any agent appearing more than expected
            agent_counts = Counter(agents_sequence)
            metrics['agent_revisits'] = sum(1 for count in agent_counts.values() if count > 2)
        else:
            metrics['immediate_agent_loops'] = 0
            metrics['agent_revisits'] = 0
        
        # Tool usage analysis
        if tool_calls:
            tool_counts = Counter(tool_calls)
            
            # Find consecutive duplicate tools
            consecutive_duplicates = 0
            for i in range(len(tool_calls) - 1):
                if tool_calls[i] == tool_calls[i + 1]:
                    consecutive_duplicates += 1
            
            metrics['consecutive_duplicate_tools'] = consecutive_duplicates
            metrics['most_used_tool'] = tool_counts.most_common(1)[0][0] if tool_counts else None
            metrics['max_tool_usage'] = tool_counts.most_common(1)[0][1] if tool_counts else 0
        else:
            metrics['consecutive_duplicate_tools'] = 0
            metrics['most_used_tool'] = None
            metrics['max_tool_usage'] = 0
        
        return metrics
    
    def detect_handoff_loops(self, agents_sequence: List[str]) -> Dict[str, Any]:
        """Detect potential loops in agent handoffs."""
        if len(agents_sequence) < 3:
            return {'has_loops': False, 'loop_patterns': []}
        
        loops = []
        
        # Check for simple A->B->A patterns
        for i in range(len(agents_sequence) - 2):
            if agents_sequence[i] == agents_sequence[i + 2]:
                pattern = f"{agents_sequence[i]} -> {agents_sequence[i+1]} -> {agents_sequence[i]}"
                loops.append({
                    'type': 'immediate_loop',
                    'pattern': pattern,
                    'position': i
                })
        
        # Check for longer cycles
        for cycle_length in range(3, min(6, len(agents_sequence) // 2)):
            for start in range(len(agents_sequence) - cycle_length * 2):
                cycle_1 = agents_sequence[start:start + cycle_length]
                cycle_2 = agents_sequence[start + cycle_length:start + cycle_length * 2]
                
                if cycle_1 == cycle_2:
                    loops.append({
                        'type': 'cycle_loop',
                        'pattern': ' -> '.join(cycle_1) + ' (repeated)',
                        'cycle_length': cycle_length,
                        'position': start
                    })
        
        return {
            'has_loops': len(loops) > 0,
            'loop_patterns': loops,
            'loop_count': len(loops)
        }
    
    def detect_duplicate_tools(self, tool_calls: List[str]) -> Dict[str, Any]:
        """Detect problematic duplicate tool usage."""
        if not tool_calls:
            return {'has_duplicates': False, 'duplicate_patterns': []}
        
        duplicates = []
        tool_positions = defaultdict(list)
        
        # Track positions of each tool
        for i, tool in enumerate(tool_calls):
            tool_positions[tool].append(i)
        
        # Find tools used multiple times
        for tool, positions in tool_positions.items():
            if len(positions) > 1:
                # Check if duplicates are consecutive (more problematic)
                consecutive_groups = []
                current_group = [positions[0]]
                
                for i in range(1, len(positions)):
                    if positions[i] == positions[i-1] + 1:
                        current_group.append(positions[i])
                    else:
                        if len(current_group) > 1:
                            consecutive_groups.append(current_group)
                        current_group = [positions[i]]
                
                if len(current_group) > 1:
                    consecutive_groups.append(current_group)
                
                duplicate_info = {
                    'tool': tool,
                    'total_usage': len(positions),
                    'positions': positions,
                    'consecutive_groups': consecutive_groups,
                    'is_problematic': len(consecutive_groups) > 0 or len(positions) > 3
                }
                duplicates.append(duplicate_info)
        
        return {
            'has_duplicates': len(duplicates) > 0,
            'duplicate_patterns': duplicates,
            'problematic_duplicates': sum(1 for d in duplicates if d['is_problematic'])
        }
