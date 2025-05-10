# app.py
import os
import json
import time
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path

# Import your existing pipeline components
from ingestion_engine import ingest
from staging_storage import stage_to_snowflake
from analysis_runner import run_analysis
from auto_story_pipeline import auto_pipeline

# Create Flask app with specific settings
app = Flask(__name__)

# Global variables to store pipeline state
pipeline_status = {
    'status': 'idle',  # idle, running, completed, error
    'logs': [],
    'charts': [],
    'story': '',
    'current_stage': '', # ingest, stage, transform, analyze, story
    'error': None
}

# Store original print function at module level
import builtins
ORIGINAL_PRINT = builtins.print

def add_log(message):
    """Add a log message to the global status"""
    pipeline_status['logs'].append(message)
    
    # Update current stage based on log content
    if 'Starting pipeline' in message:
        pipeline_status['current_stage'] = 'ingest'
    elif ('Written' in message and 'rows to' in message) or 'Staged raw table' in message:
        pipeline_status['current_stage'] = 'stage'
    elif 'Dropped rows with nulls' in message or 'Renamed columns' in message:
        pipeline_status['current_stage'] = 'transform'
    elif 'Running analysis for:' in message or 'Chart saved to' in message:
        pipeline_status['current_stage'] = 'analyze'
    elif '=== Narrative' in message:
        pipeline_status['current_stage'] = 'story'
        pipeline_status['status'] = 'completed'

def custom_auto_pipeline(url, table_hint=None):
    """Modified version of auto_pipeline that updates the global status"""
    try:
        pipeline_status['status'] = 'running'
        pipeline_status['logs'] = []
        pipeline_status['charts'] = []
        pipeline_status['story'] = ''
        pipeline_status['error'] = None
        
        # Save reference to the original print
        original_print_ref = ORIGINAL_PRINT
        
        # Define a custom print function that captures logs
        def custom_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            # Use original print reference
            original_print_ref(message, **kwargs)
            # Add to logs directly
            pipeline_status['logs'].append(message)
            
            # Update stage flag directly
            if 'Starting pipeline' in message:
                pipeline_status['current_stage'] = 'ingest'
            elif ('Written' in message and 'rows to' in message) or 'Staged raw table' in message:
                pipeline_status['current_stage'] = 'stage'
            elif 'Dropped rows with nulls' in message or 'Renamed columns' in message:
                pipeline_status['current_stage'] = 'transform'
            elif 'Running analysis for:' in message or 'Chart saved to' in message:
                pipeline_status['current_stage'] = 'analyze'
            elif '=== Narrative' in message:
                pipeline_status['current_stage'] = 'story'
                pipeline_status['status'] = 'completed'
        
        # Patch the global print function
        builtins.print = custom_print
        
        try:
            # Add the initial log message manually
            pipeline_status['logs'].append(f"Starting pipeline for URL: {url}")
            pipeline_status['current_stage'] = 'ingest'
            
            # Run the auto_pipeline
            auto_pipeline(url, table_hint)
            
        except Exception as e:
            pipeline_status['status'] = 'error'
            pipeline_status['error'] = str(e)
            pipeline_status['logs'].append(f"Error: {str(e)}")
            raise
        finally:
            # IMPORTANT: Always restore the original print
            builtins.print = original_print_ref
        
        # Gather chart information
        chart_files = list(Path("charts2").glob("*.html"))
        pipeline_status['charts'] = [f"/charts/{f.name}" for f in chart_files]
        
        # Get the story content
        story_path = Path("descriptions\\narrative.txt")
        if story_path.exists():
            pipeline_status['story'] = story_path.read_text()
        
        pipeline_status['status'] = 'completed'
        
    except Exception as e:
        pipeline_status['status'] = 'error'
        pipeline_status['error'] = str(e)
        pipeline_status['logs'].append(f"Error: {str(e)}")
        
        # Safety: ensure print is restored
        builtins.print = ORIGINAL_PRINT

def run_custom_analysis(request_text):
    """Run a specific analysis and update the status"""
    try:
        pipeline_status['status'] = 'running'
        pipeline_status['logs'] = [f"Running analysis: {request_text}"]
        pipeline_status['current_stage'] = 'analyze'
        
        # Save reference to original print
        original_print_ref = ORIGINAL_PRINT
        
        # Define a custom print function that captures logs
        def custom_print(*args, **kwargs):
            message = " ".join(str(arg) for arg in args)
            # Use original reference
            original_print_ref(message, **kwargs)
            # Add to logs directly
            pipeline_status['logs'].append(message)
        
        # Patch the global print function
        builtins.print = custom_print
        
        try:
            # Run the analysis
            result = run_analysis(request_text)
        except Exception as e:
            pipeline_status['status'] = 'error'
            pipeline_status['error'] = str(e)
            pipeline_status['logs'].append(f"Error: {str(e)}")
            raise
        finally:
            # Always restore the original print
            builtins.print = original_print_ref
        
        # Update charts list
        chart_files = list(Path("charts2").glob("*.html"))
        pipeline_status['charts'] = [f"/charts/{f.name}" for f in chart_files]
        
        pipeline_status['status'] = 'completed'
        return result
    
    except Exception as e:
        pipeline_status['status'] = 'error'
        pipeline_status['error'] = str(e)
        pipeline_status['logs'].append(f"Error: {str(e)}")
        
        # Safety: ensure print is restored
        builtins.print = ORIGINAL_PRINT
        return f"Error: {str(e)}"

# Create necessary directories
for d in ["charts2", "charts2/json", "public", "descriptions", "codes", "prompts", "responses"]:
    os.makedirs(d, exist_ok=True)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ingest', methods=['POST'])
def api_ingest():
    """Start the pipeline with the given URL"""
    data = request.json
    url = data.get('url')
    table_hint = data.get('table_hint')
    
    if not url:
        return jsonify({'error': 'URL is required'}), 400
    
    # Start the pipeline in a background thread
    thread = threading.Thread(target=custom_auto_pipeline, args=(url, table_hint))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Pipeline started'})

@app.route('/api/analysis', methods=['POST'])
def api_analysis():
    """Run a custom analysis"""
    data = request.json
    analysis_request = data.get('request')
    
    if not analysis_request:
        return jsonify({'error': 'Analysis request is required'}), 400
    
    if not Path(".last_table").exists():
        return jsonify({'error': 'No table loaded. Please ingest data first.'}), 400
    
    # Run the analysis in a background thread
    thread = threading.Thread(target=run_custom_analysis, args=(analysis_request,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'Analysis started'})

@app.route('/api/status')
def api_status():
    """Get the current pipeline status"""
    return jsonify(pipeline_status)

@app.route('/charts/<path:filename>')
def serve_chart(filename):
    """Serve chart files"""
    return send_from_directory('charts2', filename)

# Add a run script that explicitly disables reloading
if __name__ == '__main__':
    # CRITICAL FIX: Use the no-reload option to prevent Flask from restarting 
    # when charts are saved to the charts2 directory
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)