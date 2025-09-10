#!/bin/bash
set -e

echo "🚂 Starting Historical Photos API on Railway..."

# Function to check if environment variable is set
check_env_var() {
    local var_name="$1"
    local var_value="${!var_name}"
    
    if [ -z "$var_value" ]; then
        echo "❌ ERROR: Environment variable $var_name is not set!"
        echo "💡 Set it in Railway dashboard: Variables section"
        return 1
    else
        echo "✅ $var_name is configured"
        return 0
    fi
}

# Railway-specific environment validation
echo "🔍 Checking Railway environment variables..."

check_env_var "PINECONE_API_KEY"
PINECONE_OK=$?

# Optional variables (warn but don't fail)
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY not set - Gemini reranking disabled"
fi

if [ -z "$GROQ_API_KEY" ]; then
    echo "⚠️  WARNING: GROQ_API_KEY not set - Groq features disabled"
fi

# Exit if critical variables are missing
if [ $PINECONE_OK -ne 0 ]; then
    echo ""
    echo "🚂 Railway Setup Instructions:"
    echo "   1. Go to your Railway project dashboard"
    echo "   2. Click on 'Variables' tab"
    echo "   3. Add these environment variables:"
    echo "      PINECONE_API_KEY = your_pinecone_key_here"
    echo "      GEMINI_API_KEY = your_gemini_key_here (optional)"
    echo "      GROQ_API_KEY = your_groq_key_here (optional)"
    exit 1
fi

# Railway-specific configuration
export FLASK_ENV="${FLASK_ENV:-production}"
export FLASK_DEBUG="${FLASK_DEBUG:-false}"
export PORT="${PORT:-5000}"  # Railway sets PORT automatically
export PYTHONPATH="${PYTHONPATH:-/app:/app/main_app}"

# Create necessary directories
mkdir -p /app/logs /app/main_app_outputs

# Railway system information
echo "🚂 Railway Configuration:"
echo "   Flask Environment: $FLASK_ENV"
echo "   Flask Debug: $FLASK_DEBUG"
echo "   Port: $PORT"
echo "   Python Path: $PYTHONPATH"

# System resources check (8GB RAM available)
echo "💾 System Resources:"
if command -v free >/dev/null 2>&1; then
    TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    AVAIL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    echo "   Total RAM: ${TOTAL_MEM}MB"
    echo "   Available: ${AVAIL_MEM}MB"
    echo "   🎯 Railway Plan: 8GB RAM perfect for this app!"
else
    echo "   ℹ️  Running on Railway cloud platform"
fi

if [ -n "$RAILWAY_ENVIRONMENT" ]; then
    echo "   Environment: $RAILWAY_ENVIRONMENT"
fi

# Test Pinecone connection
echo "🔌 Testing Pinecone connection..."
python3 -c "
import os
try:
    from pinecone import Pinecone
    client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    indexes = [idx.name for idx in client.list_indexes()]
    print(f'✅ Pinecone connected successfully!')
    print(f'📊 Available indexes: {indexes}')
except Exception as e:
    print(f'⚠️  Pinecone connection warning: {e}')
    print('🔄 Will retry during app initialization...')
" || echo "⚠️  Pinecone test failed - will retry during startup"

# Handle shutdown gracefully for Railway
trap 'echo "🛑 Railway shutdown signal received..."; kill -TERM $PID; wait $PID' TERM INT

echo ""
echo "🌟 All checks passed! Starting Flask application on Railway..."
echo "📡 API will be available at: https://your-railway-domain/"
echo "🏥 Health check: https://your-railway-domain/health"
echo "🔍 API status: https://your-railway-domain/api/status"
echo "💰 Railway usage: $5 minimum, pay-per-use scaling"

# Start the application in background
exec "$@" &
PID=$!
wait $PID