#!/bin/bash

echo "Starting CryptoScriptHub..."

# Kill any existing processes
pkill -f "python.*app\.py" 2>/dev/null
pkill -f "vite" 2>/dev/null
sleep 2

# Start backend
echo "Starting backend on port 5005..."
cd /workspace/CryptoScriptHub
python backend/app.py > backend.log 2>&1 &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend on port 12005..."
cd crypto-frontend
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 3

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Services started!"
echo "Frontend: http://localhost:12005"
echo "Backend API: http://localhost:5005"
echo ""
echo "Testing backend health..."
curl -s http://localhost:5005/api/health | jq . || echo "Backend not responding"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup processes on exit
cleanup() {
    echo "Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait