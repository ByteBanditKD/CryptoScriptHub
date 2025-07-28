from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import os
import json
import tempfile
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import threading
import time
import uuid

app = Flask(__name__)
CORS(app)

# Define the root directory
ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
IMG_DIR = ROOT_DIR / "img"

# Ensure directories exist
RESULTS_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# Store running jobs
running_jobs = {}

# Define available scripts
SCRIPTS = {
    "support-resistance": {
        "CryptoSRPredictor_CLI.py": {
            "name": "Support & Resistance Predictor",
            "description": "Predict support and resistance levels using ML",
            "params": [
                {"name": "symbol", "type": "text", "default": "BTC/USDT", "required": True},
                {"name": "timeframe", "type": "select", "options": ["1h", "4h", "1d"], "default": "1d"},
                {"name": "limit", "type": "number", "default": 100, "min": 50, "max": 1000}
            ]
        },
        "Sup_Res_Finder.py": {
            "name": "Support & Resistance Finder",
            "description": "Find key support and resistance levels",
            "params": [
                {"name": "symbol", "type": "text", "default": "BTCUSDT", "required": True}
            ]
        }
    },
    "scanners": {
        "OversoldScanner.py": {
            "name": "Oversold Scanner",
            "description": "Scan for oversold cryptocurrencies (RSI < 30)",
            "params": [
                {"name": "scan_type", "type": "select", "options": ["all", "custom"], "default": "all"},
                {"name": "symbols", "type": "textarea", "default": "BTCUSDT,ETHUSDT,XRPUSDT", "required": False}
            ]
        },
        "HeadAndShouldersScanner.py": {
            "name": "Head & Shoulders Scanner",
            "description": "Scan for head and shoulders patterns",
            "params": []
        },
        "TriangleScanner.py": {
            "name": "Triangle Pattern Scanner",
            "description": "Scan for triangle chart patterns",
            "params": []
        },
        "TD_Sequential_Scanner.py": {
            "name": "TD Sequential Scanner",
            "description": "Scan using TD Sequential indicator",
            "params": []
        }
    },
    "price-prediction": {
        "CryptoPricePredictor.py": {
            "name": "Crypto Price Predictor",
            "description": "Predict future prices using LSTM model",
            "params": [
                {"name": "symbol", "type": "text", "default": "BTC", "required": True},
                {"name": "days", "type": "number", "default": 7, "min": 1, "max": 30}
            ]
        },
        "Price_Prediction_Classifier.py": {
            "name": "Price Direction Classifier",
            "description": "Classify price direction (up/down)",
            "params": [
                {"name": "symbol", "type": "text", "default": "BTCUSDT", "required": True}
            ]
        }
    },
    "trend-analysis": {
        "CryptoTA_API.py": {
            "name": "Crypto Technical Analysis",
            "description": "Comprehensive technical analysis with multiple indicators",
            "params": [
                {"name": "symbol", "type": "text", "default": "BTC", "required": True},
                {"name": "timeframe", "type": "select", "options": ["1h", "4h", "1d"], "default": "1d"}
            ]
        },
        "CryptoTA_Multiple.py": {
            "name": "Multi-Symbol Technical Analysis",
            "description": "Technical analysis for multiple symbols",
            "params": [
                {"name": "symbols", "type": "textarea", "default": "BTC,ETH,XRP", "required": True}
            ]
        }
    }
}

@app.route('/api/scripts', methods=['GET'])
def get_scripts():
    """Get all available scripts organized by category"""
    return jsonify(SCRIPTS)

@app.route('/api/run-script', methods=['POST'])
def run_script():
    """Run a script with given parameters"""
    data = request.json
    category = data.get('category')
    script_name = data.get('script')
    params = data.get('params', {})
    
    if not category or not script_name:
        return jsonify({"error": "Category and script name are required"}), 400
    
    if category not in SCRIPTS or script_name not in SCRIPTS[category]:
        return jsonify({"error": "Script not found"}), 404
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Start script execution in background
    thread = threading.Thread(target=execute_script, args=(job_id, category, script_name, params))
    thread.start()
    
    return jsonify({"job_id": job_id, "status": "started"})

def execute_script(job_id, category, script_name, params):
    """Execute script in background"""
    running_jobs[job_id] = {
        "status": "running",
        "output": "",
        "error": "",
        "start_time": datetime.now(),
        "results": []
    }
    
    try:
        script_path = ROOT_DIR / category / script_name
        
        # Change to script directory
        script_dir = script_path.parent
        
        # Prepare command
        cmd = ["python", str(script_path)]
        
        # For scripts that need interactive input, we'll modify them or use environment variables
        env = os.environ.copy()
        
        # Set parameters as environment variables for scripts to read
        for key, value in params.items():
            env[f"CRYPTO_PARAM_{key.upper()}"] = str(value)
        
        # Execute script
        process = subprocess.Popen(
            cmd,
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        stdout, stderr = process.communicate(timeout=300)  # 5 minute timeout
        
        running_jobs[job_id]["output"] = stdout
        running_jobs[job_id]["error"] = stderr
        running_jobs[job_id]["status"] = "completed" if process.returncode == 0 else "failed"
        
        # Look for result files
        result_files = []
        
        # Check for CSV results
        for csv_file in RESULTS_DIR.glob("*.csv"):
            if csv_file.stat().st_mtime > running_jobs[job_id]["start_time"].timestamp():
                result_files.append({
                    "type": "csv",
                    "filename": csv_file.name,
                    "path": str(csv_file)
                })
        
        # Check for JSON results
        for json_file in RESULTS_DIR.glob("*.json"):
            if json_file.stat().st_mtime > running_jobs[job_id]["start_time"].timestamp():
                result_files.append({
                    "type": "json",
                    "filename": json_file.name,
                    "path": str(json_file)
                })
        
        # Check for image results
        for img_file in IMG_DIR.glob("*.png"):
            if img_file.stat().st_mtime > running_jobs[job_id]["start_time"].timestamp():
                result_files.append({
                    "type": "image",
                    "filename": img_file.name,
                    "path": str(img_file)
                })
        
        running_jobs[job_id]["results"] = result_files
        
    except subprocess.TimeoutExpired:
        running_jobs[job_id]["status"] = "timeout"
        running_jobs[job_id]["error"] = "Script execution timed out"
    except Exception as e:
        running_jobs[job_id]["status"] = "failed"
        running_jobs[job_id]["error"] = str(e)

@app.route('/api/job-status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get status of a running job"""
    if job_id not in running_jobs:
        return jsonify({"error": "Job not found"}), 404
    
    job = running_jobs[job_id].copy()
    job["start_time"] = job["start_time"].isoformat()
    
    return jsonify(job)

@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download result files"""
    # Check in results directory
    file_path = RESULTS_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    
    # Check in images directory
    file_path = IMG_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404

@app.route('/api/csv-data/<filename>', methods=['GET'])
def get_csv_data(filename):
    """Get CSV data as JSON for display"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    try:
        df = pd.read_csv(file_path)
        # Convert to JSON with proper handling of NaN values
        data = df.fillna("").to_dict('records')
        columns = df.columns.tolist()
        
        return jsonify({
            "columns": columns,
            "data": data,
            "total_rows": len(data)
        })
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {str(e)}"}), 500

@app.route('/api/json-data/<filename>', methods=['GET'])
def get_json_data(filename):
    """Get JSON data for display"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": f"Error reading JSON: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    print("Starting CryptoScriptHub backend...")
    app.run(host='0.0.0.0', port=5005, debug=False, use_reloader=False)