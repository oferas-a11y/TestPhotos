#!/bin/bash
set -e

echo "🚀 Starting Historical Photos API..."

# Function to check if environment variable is set
check_env_var() {
    local var_name="$1"
    local var_value="${!var_name}"
    
    if [ -z "$var_value" ]; then
        echo "❌ ERROR: Environment variable $var_name is not set!"
        echo "Please set it in your .env file or docker-compose.yml"
        return 1
    else
        echo "✅ $var_name is configured"
        return 0
    fi
}

# Validate critical environment variables
echo "🔍 Checking environment variables..."

check_env_var "PINECONE_API_KEY"
PINECONE_OK=$?

# Optional variables (warn but don't fail)
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY not set - Gemini reranking will be disabled"
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY not set - Groq features will be disabled"
fi

# Exit if critical variables are missing
if [ $PINECONE_OK -ne 0 ]; then
    echo "💡 Fix: Add your API keys to .env file:"
    echo "   PINECONE_API_KEY=your_pinecone_key_here"
    echo "   GEMINI_API_KEY=your_gemini_key_here"
    echo "   GROQ_API_KEY=your_groq_key_here"
    exit 1
fi

# Set default values
export FLASK_ENV="${FLASK_ENV:-production}"
export FLASK_DEBUG="${FLASK_DEBUG:-false}"
export PORT="${PORT:-5000}"
export PYTHONPATH="${PYTHONPATH:-/app:/app/main_app}"

# Create necessary directories
mkdir -p /app/logs /app/main_app_outputs

echo "🔧 Configuration:"
echo "   Flask Environment: $FLASK_ENV"
echo "   Flask Debug: $FLASK_DEBUG"
echo "   Port: $PORT"
echo "   Python Path: $PYTHONPATH"

# Memory monitoring for VPS
echo "💾 Memory Status:"
if command -v free >/dev/null 2>&1; then
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    AVAIL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    echo "   Total RAM: ${TOTAL_MEM}MB"
    echo "   Available: ${AVAIL_MEM}MB"
    
    if [ "$AVAIL_MEM" -lt 800 ]; then
        echo "   ⚠️  WARNING: Low memory detected (${AVAIL_MEM}MB available)"
        echo "   💡 Consider restarting if app fails to initialize"
    else
        echo "   ✅ Memory looks good for app startup (${AVAIL_MEM}MB available)"
    fi
else
    echo "   ℹ️  Memory monitoring not available"
fi

# Test Pinecone connection (optional quick check)
echo "🔌 Testing Pinecone connection..."
python3 -c "
import os
try:
    from pinecone import Pinecone
    client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    indexes = [idx.name for idx in client.list_indexes()]
    print(f'✅ Pinecone connected successfully. Available indexes: {indexes}')
except Exception as e:
    print(f'⚠️  Pinecone connection warning: {e}')
    print('🔄 Will retry during app initialization...')
" || echo "⚠️  Pinecone test failed - will retry during startup"

# Handle shutdown gracefully
trap 'echo "🛑 Shutting down gracefully..."; kill -TERM $PID; wait $PID' TERM INT

echo "🌟 All checks passed! Starting Flask application..."
echo "📡 API will be available at: http://localhost:$PORT"
echo "🏥 Health check endpoint: http://localhost:$PORT/health"
echo "🔍 API status endpoint: http://localhost:$PORT/api/status"

# Start the application
exec "$@" &
PID=$!
wait $PID