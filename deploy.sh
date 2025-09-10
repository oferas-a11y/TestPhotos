#!/bin/bash

# Historical Photos API - VPS Deployment Script
# Usage: ./deploy.sh [build|deploy|restart|logs|status]

set -e

# Configuration
APP_NAME="historical-photos-api"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed!"
        log_info "Install Docker: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed!"
        log_info "Install Docker Compose: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    log_success "Docker and Docker Compose are available"
}

# Check environment file
check_env() {
    if [ ! -f "$ENV_FILE" ]; then
        log_warning ".env file not found. Creating from template..."
        
        cat > "$ENV_FILE" << 'EOF'
# Historical Photos API Environment Variables
# Copy your API keys here

# Required: Pinecone Vector Database
PINECONE_API_KEY=your_pinecone_api_key_here

# Optional: AI Enhancement Services
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Flask Configuration (usually don't need to change)
FLASK_ENV=production
FLASK_DEBUG=false
PORT=5000
EOF
        
        log_warning "Created .env file with template. Please add your API keys!"
        log_info "Get Pinecone key: https://www.pinecone.io/"
        log_info "Edit .env file and run deploy again"
        return 1
    fi
    
    # Check if required keys are set
    if grep -q "your_pinecone_api_key_here" "$ENV_FILE"; then
        log_error "Please set your PINECONE_API_KEY in .env file"
        return 1
    fi
    
    log_success "Environment configuration found"
    return 0
}

# Build Docker image
build_app() {
    log_info "Building Docker image..."
    
    if [ -f "requirements-docker.txt" ]; then
        log_info "Using production requirements (requirements-docker.txt)"
        cp requirements-docker.txt requirements.txt.bak
        cp requirements-docker.txt requirements.txt
    fi
    
    docker-compose build --no-cache
    
    if [ -f "requirements.txt.bak" ]; then
        mv requirements.txt.bak requirements.txt
    fi
    
    log_success "Docker image built successfully"
}

# Deploy application
deploy_app() {
    log_info "Deploying $APP_NAME..."
    
    # Stop existing containers
    docker-compose down --remove-orphans 2>/dev/null || true
    
    # Start new containers
    docker-compose up -d
    
    log_info "Waiting for application to start..."
    sleep 10
    
    # Check health
    check_health
    
    log_success "Deployment completed!"
    log_info "API is available at: http://localhost:5000"
    log_info "Health check: http://localhost:5000/health"
}

# Check application health
check_health() {
    local max_attempts=12
    local attempt=1
    
    log_info "Checking application health..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
            log_success "Application is healthy!"
            return 0
        fi
        
        log_info "Attempt $attempt/$max_attempts - waiting for app to start..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    log_error "Application failed health check!"
    log_info "Check logs: ./deploy.sh logs"
    return 1
}

# Show application logs
show_logs() {
    log_info "Showing application logs (press Ctrl+C to exit)..."
    docker-compose logs -f --tail=50 testphotos-api
}

# Show application status
show_status() {
    log_info "Application Status:"
    echo "===================="
    
    # Container status
    docker-compose ps
    echo
    
    # Health check
    if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
        log_success "✅ Application is running and healthy"
        
        # Try to get API status
        if curl -f -s http://localhost:5000/api/status > /dev/null 2>&1; then
            log_success "✅ API endpoints are responding"
            echo
            log_info "API Status:"
            curl -s http://localhost:5000/api/status | python3 -m json.tool 2>/dev/null || echo "API status not available"
        else
            log_warning "⚠️  API endpoints not responding (may be initializing)"
        fi
    else
        log_error "❌ Application health check failed"
    fi
    
    echo
    log_info "Quick Commands:"
    echo "  Health Check: curl http://localhost:5000/health"
    echo "  API Status:   curl http://localhost:5000/api/status"
    echo "  Semantic Search: curl 'http://localhost:5000/api/search/semantic?q=love&limit=5'"
}

# Restart application
restart_app() {
    log_info "Restarting $APP_NAME..."
    docker-compose restart testphotos-api
    sleep 5
    check_health
    log_success "Application restarted!"
}

# Stop application
stop_app() {
    log_info "Stopping $APP_NAME..."
    docker-compose down
    log_success "Application stopped!"
}

# Update application
update_app() {
    log_info "Updating $APP_NAME..."
    
    # Pull latest code (if using git)
    if [ -d ".git" ]; then
        log_info "Pulling latest code..."
        git pull
    fi
    
    # Rebuild and deploy
    build_app
    deploy_app
    
    log_success "Application updated!"
}

# Main script logic
main() {
    local command="${1:-help}"
    
    echo "🌟 Historical Photos API Deployment Script"
    echo "============================================="
    
    case $command in
        "build")
            check_docker
            build_app
            ;;
        "deploy")
            check_docker
            if check_env; then
                build_app
                deploy_app
            fi
            ;;
        "restart")
            check_docker
            restart_app
            ;;
        "logs")
            show_logs
            ;;
        "status")
            show_status
            ;;
        "stop")
            stop_app
            ;;
        "update")
            check_docker
            if check_env; then
                update_app
            fi
            ;;
        "help"|*)
            echo "Usage: $0 [command]"
            echo
            echo "Commands:"
            echo "  build    - Build Docker image only"
            echo "  deploy   - Build and deploy application"
            echo "  restart  - Restart running application"
            echo "  logs     - Show application logs"
            echo "  status   - Show application status and health"
            echo "  stop     - Stop application"
            echo "  update   - Pull code and redeploy"
            echo "  help     - Show this help message"
            echo
            echo "Examples:"
            echo "  $0 deploy   # First-time deployment"
            echo "  $0 status   # Check if app is running"
            echo "  $0 logs     # View logs"
            echo "  $0 restart  # Restart after changes"
            ;;
    esac
}

# Run main function with all arguments
main "$@"