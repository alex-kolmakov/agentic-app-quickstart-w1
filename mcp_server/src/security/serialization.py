import json
import base64
import io
from typing import Any, Dict
import matplotlib.pyplot as plt

class ExecutionResultSerializer:
    def serialize_execution_result(self, code: str, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize execution results including plots."""
        serialized = {
            'code': code,
            'status': execution_result.get('status', 'success'),
            'result': None,
            'plots': [],
            'error': execution_result.get('message') if execution_result.get('status') == 'error' else None
        }
        
        # Handle the result
        result = execution_result.get('result')
        if result is not None:
            try:
                # Try to convert pandas/numpy objects to JSON-serializable format
                if hasattr(result, 'to_dict'):
                    serialized['result'] = result.to_dict()
                elif hasattr(result, 'tolist'):
                    serialized['result'] = result.tolist()
                elif hasattr(result, '__str__'):
                    serialized['result'] = str(result)
                else:
                    serialized['result'] = result
            except:
                serialized['result'] = str(result)
        
        # Handle matplotlib plots
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                
                # Save plot to base64 string
                buffer = io.BytesIO()
                fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
                buffer.seek(0)
                
                plot_data = base64.b64encode(buffer.getvalue()).decode()
                serialized['plots'].append({
                    'figure_id': fig_num,
                    'data': plot_data,
                    'format': 'png'
                })
                
                buffer.close()
                plt.close(fig)
        
        return serialized
