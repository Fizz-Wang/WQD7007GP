# dashboard_app.py

import subprocess
import os
import json
import threading
import time
from flask import Flask, render_template, request, jsonify

# Import our project's HBase connection utility
from utils.hbase_connector import get_hbase_connection

# Initialize Flask App
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-should-be-changed'

# --- [NEW] Define and register a custom Jinja2 filter ---
def from_json_filter(json_string):
    """Custom filter to parse a JSON string into a Python object."""
    if not json_string:
        return None
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        # Return an empty dict or None if parsing fails
        return {}

app.jinja_env.filters['fromjson'] = from_json_filter


# --- [Page Route] Home / Task Console ---
@app.route('/')
def index():
    """Renders the main page, the Task Console."""
    return render_template('index.html')

# --- [API Route] Run Spark Task in the Background ---
@app.route('/run-task/<task_id>', methods=['POST'])
def run_task(task_id):
    """
    This route now only triggers the Spark job in a background thread
    and immediately returns a success response. It no longer streams logs.
    """
    # Create a unique log file for each task run
    log_file_path = f"/tmp/spark_task_{task_id}_{int(time.time())}.log"
    
    def run_spark_in_background():
        project_root = os.getcwd()
        command = ['./run_spark.sh', task_id]
        
        # The output of the Spark job is now written to a dedicated log file
        with open(log_file_path, 'w') as log_file:
            subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=project_root
            )
    
    # Start the Spark job in a separate thread
    thread = threading.Thread(target=run_spark_in_background)
    thread.start()
    
    # Immediately return a success message to the user
    # The user can then monitor the log file manually via SSH
    return jsonify({
        'status': 'success', 
        'message': f'Task {task_id} has been started in the background.',
        'log_command': f'tail -f {log_file_path}'
    })


# --- [Page Route] Model Registry and Comparison ---
@app.route('/model-registry')
def model_registry():
    """Renders the upgraded Model Registry, now a Comparison Workbench."""
    model_data = []
    error_message = None

    connection = get_hbase_connection()
    if not connection:
        error_message = "Could not connect to HBase. Please ensure services are running."
    else:
        try:
            table = connection.table('model_registry')
            for key, data in table.scan():
                item = {key.decode('utf-8'): val.decode('utf-8') for key, val in data.items()}
                item['row_key'] = key.decode('utf-8')
                model_data.append(item)
            
            # Get filter parameters from the URL
            task_filter = request.args.get('task')
            model_filter = request.args.get('model')
            sort_by = request.args.get('sort', 'info:run_timestamp')
            sort_order = request.args.get('order', 'desc')

            # Create lists for dropdown menus
            unique_tasks = sorted(list(set(d.get('info:task_name', 'N/A') for d in model_data)))
            unique_models = sorted(list(set(d.get('info:model_name', 'N/A') for d in model_data)))

            # Apply filters
            if task_filter:
                model_data = [d for d in model_data if d.get('info:task_name') == task_filter]
            if model_filter:
                model_data = [d for d in model_data if d.get('info:model_name') == model_filter]
            
            # Apply sorting
            is_numeric_sort = 'evaluation_score' in sort_by or 'training_time' in sort_by
            model_data.sort(
                key=lambda x: float(x.get(sort_by, 0)) if is_numeric_sort else x.get(sort_by, ''),
                reverse=(sort_order == 'desc')
            )
        except Exception as e:
            error_message = f"An error occurred while fetching data from HBase: {e}"
        finally:
            connection.close()

    return render_template('model_registry.html', 
                           data=model_data, 
                           error=error_message,
                           unique_tasks=unique_tasks,
                           unique_models=unique_models,
                           current_filters={'task': task_filter, 'model': model_filter},
                           current_sort={'by': sort_by, 'order': sort_order})


# --- [Page Route] Experiment Details and Visualization ---
@app.route('/experiment/<row_key>')
def experiment_details(row_key):
    """
    Renders a new page showing detailed metrics and visualizations
    for a single experiment, identified by its HBase RowKey.
    """
    data = {}
    error_message = None
    
    connection = get_hbase_connection()
    if not connection:
        error_message = "Could not connect to HBase."
    else:
        try:
            table = connection.table('model_registry')
            row_data = table.row(row_key.encode('utf-8'))
            if row_data:
                data = {key.decode('utf-8'): val.decode('utf-8') for key, val in row_data.items()}
                data['row_key'] = row_key
            else:
                error_message = f"Experiment with ID '{row_key}' not found."
        except Exception as e:
            error_message = f"An error occurred while fetching experiment details: {e}"
        finally:
            connection.close()
            
    return render_template('experiment_details.html', data=data, error=error_message)

# --- Main Application Runner ---
if __name__ == '__main__':
    # threaded=True is important for handling multiple requests concurrently
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

