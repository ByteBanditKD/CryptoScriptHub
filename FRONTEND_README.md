# CryptoScriptHub React Frontend

A modern, user-friendly React frontend for running cryptocurrency analysis scripts from the CryptoScriptHub collection.

## 🚀 Features

- **Modern UI**: Dark theme with Material-UI components
- **Script Categories**: Organized by functionality (Support/Resistance, Scanners, Price Prediction, Trend Analysis)
- **Real-time Execution**: Run scripts with live status updates
- **Parameter Configuration**: Easy-to-use forms for script parameters
- **Results Visualization**: View CSV data in tables and download files
- **Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
CryptoScriptHub/
├── crypto-frontend/          # React frontend application
│   ├── src/
│   │   ├── components/       # React components
│   │   │   ├── Dashboard.jsx # Main dashboard with script categories
│   │   │   ├── ScriptRunner.jsx # Script execution interface
│   │   │   ├── Results.jsx   # Results viewing page
│   │   │   └── Navbar.jsx    # Navigation bar
│   │   ├── App.jsx          # Main app component
│   │   └── App.css          # Styling
│   └── package.json         # Frontend dependencies
├── backend/                  # Flask backend API
│   ├── app.py               # Main Flask application
│   └── requirements.txt     # Backend dependencies
├── start.sh                 # Startup script for both services
└── FRONTEND_README.md       # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Node.js (v16 or higher)
- Python 3.7+
- pip (Python package manager)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/ByteBanditKD/CryptoScriptHub.git
   cd CryptoScriptHub
   ```

2. **Install backend dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. **Install frontend dependencies**:
   ```bash
   cd crypto-frontend
   npm install
   cd ..
   ```

4. **Start both services**:
   ```bash
   chmod +x start.sh
   ./start.sh
   ```

5. **Access the application**:
   - Frontend: http://localhost:12006 (or check startup logs for actual port)
   - Backend API: http://localhost:5005
   
   **Note:** The frontend may use a different port if the default is occupied. Check the startup script output for the actual port.

### Manual Setup (Alternative)

If you prefer to run services separately:

**Terminal 1 - Backend:**
```bash
cd CryptoScriptHub
python backend/app.py
```

**Terminal 2 - Frontend:**
```bash
cd CryptoScriptHub/crypto-frontend
npm run dev
```

## 🎯 How to Use

### 1. Dashboard
- View all available script categories
- Each category shows the number of available scripts
- Click "Explore Scripts" to access scripts in that category

### 2. Running Scripts
- Select a script category from the dashboard
- Choose a specific script to run
- Configure parameters using the form fields:
  - **Text fields**: Enter symbols (e.g., BTC, ETH)
  - **Select dropdowns**: Choose timeframes or options
  - **Number fields**: Set limits or days
  - **Text areas**: Enter multiple symbols (comma-separated)
- Click "Run Script" to execute
- Monitor progress in the status dialog

### 3. Viewing Results
- Results appear automatically after script completion
- Download CSV files or images directly
- View CSV data in formatted tables
- Access the Results page for all generated files

## 📊 Available Script Categories

### Support & Resistance
- **Support & Resistance Predictor**: ML-based level prediction
- **Support & Resistance Finder**: Key level identification

### Scanners
- **Oversold Scanner**: Find oversold cryptocurrencies (RSI < 30)
- **Head & Shoulders Scanner**: Pattern recognition
- **Triangle Pattern Scanner**: Geometric pattern detection
- **TD Sequential Scanner**: TD Sequential indicator analysis

### Price Prediction
- **Crypto Price Predictor**: LSTM-based price forecasting
- **Price Direction Classifier**: Up/down movement classification

### Trend Analysis
- **Crypto Technical Analysis**: Comprehensive indicator analysis
- **Multi-Symbol Technical Analysis**: Batch analysis for multiple symbols

## 🔧 Configuration

### Backend Configuration
The Flask backend runs on port 5000 and provides these endpoints:
- `GET /api/scripts` - Get available scripts
- `POST /api/run-script` - Execute a script
- `GET /api/job-status/<job_id>` - Check execution status
- `GET /api/download/<filename>` - Download result files
- `GET /api/csv-data/<filename>` - Get CSV data as JSON
- `GET /api/json-data/<filename>` - Get JSON analysis results

### Frontend Configuration
The React frontend runs on port 12000 with:
- Proxy configuration for API calls
- Material-UI dark theme
- Responsive design breakpoints

## 📝 Script Parameters

Scripts accept various parameter types:
- **symbol**: Cryptocurrency symbol (BTC, ETH, etc.)
- **timeframe**: Chart timeframe (1h, 4h, 1d)
- **limit**: Number of data points to analyze
- **symbols**: Multiple symbols (comma-separated)
- **days**: Number of days for prediction

## 📁 Results & Output

Scripts generate various output formats:
- **CSV files**: Tabular data results
- **JSON files**: Structured analysis results
- **PNG images**: Charts and visualizations
- **Console output**: Real-time execution logs

Results are stored in:
- `results/` - CSV and JSON files
- `img/` - Generated charts and images

## 🚨 Important Notes

### Disclaimer
These scripts are educational examples only. They are not intended for use in live trading without extensive testing, validation, and professional review. Price predictions are based on historical data and machine learning models, which do not guarantee future results. Use at your own risk and do not rely on these scripts for real financial decisions.

### Rate Limits
- Scripts use public APIs from cryptocurrency exchanges
- Some exchanges have rate limits - scripts include built-in delays
- If you encounter rate limit errors, wait a few minutes before retrying

### Dependencies
- Scripts require internet connection for fetching market data
- Some ML models may take time to train on first run
- Ensure all Python dependencies are installed correctly

## 🐛 Troubleshooting

### Common Issues

1. **Backend won't start**:
   - Check if all Python dependencies are installed
   - Ensure port 5000 is not in use
   - Check Python version (3.7+ required)

2. **Frontend won't start**:
   - Run `npm install` in the crypto-frontend directory
   - Check if port 12000 is available
   - Ensure Node.js version is 16+

3. **Scripts fail to run**:
   - Check internet connection
   - Verify cryptocurrency symbols are valid
   - Check console output for specific errors

4. **No results generated**:
   - Scripts may take time to complete
   - Check the results/ and img/ directories
   - Some scripts only generate console output

### Getting Help

If you encounter issues:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure you have a stable internet connection
4. Check the original script documentation in the repository

## 🔄 Development

### Adding New Scripts

1. Add the script to the appropriate category folder
2. Update `backend/app.py` SCRIPTS dictionary
3. Ensure the script can read parameters from environment variables
4. Test the script integration

### Customizing the UI

- Modify components in `crypto-frontend/src/components/`
- Update styling in `crypto-frontend/src/App.css`
- Adjust theme in `crypto-frontend/src/App.jsx`

## 📄 License

This project follows the same license as the original CryptoScriptHub repository.