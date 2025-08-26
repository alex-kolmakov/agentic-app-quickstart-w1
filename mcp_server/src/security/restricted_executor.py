from typing import Dict, Any

class RestrictedPythonExecutor:
    def __init__(self, allowed_globals=None):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path
        import os
        import datetime
        import math
        import statistics
        
        self.allowed_globals = allowed_globals or {
            # Built-in functions
            '__builtins__': {
                'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
                'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round,
                'sorted': sorted, 'enumerate': enumerate, 'range': range, 'zip': zip,
                'map': map, 'filter': filter, 'any': any, 'all': all
            },
            # Data analysis libraries
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'Path': Path,
            'os': os,
            'datetime': datetime,
            'math': math,
            'statistics': statistics,
        }

    def execute(self, code_string: str, extra_globals: Dict[str, Any] = None) -> Dict[str, Any]:
        execution_globals = self.allowed_globals.copy()
        if extra_globals:
            execution_globals.update(extra_globals)
            
        result = None
        try:
            # Execute the code
            exec(code_string, execution_globals)
            
            # Try to get the result from the last expression
            lines = code_string.strip().split('\n')
            if lines:
                last_line = lines[-1].strip()
                if last_line and not last_line.startswith('#') and '=' not in last_line:
                    try:
                        result = eval(last_line, execution_globals)
                    except:
                        pass
                        
        except Exception as e:
            return {'status': 'error', 'message': str(e), 'result': None}
            
        return {'status': 'success', 'result': result, 'globals': execution_globals}
