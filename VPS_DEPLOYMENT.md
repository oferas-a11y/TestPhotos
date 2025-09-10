# 🚀 VPS Deployment Guide - Historical Photos API

Complete guide for deploying your Historical Photos API on any VPS (RamNode, DigitalOcean, Linode, etc.) using Docker.

## 📋 Prerequisites

### VPS Requirements
- **RAM:** 2GB minimum (4GB recommended)
- **CPU:** 1 vCPU minimum 
- **Storage:** 10GB minimum
- **OS:** Ubuntu 20.04+ or any Docker-compatible Linux

### Software Requirements
- Docker & Docker Compose
- Git (for cloning repository)
- curl (for testing)

## 🛠️ VPS Setup

### 1. Connect to Your VPS
```bash
ssh root@your-vps-ip
# or
ssh username@your-vps-ip
```

### 2. Install Docker
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Add user to docker group (optional)
sudo usermod -aG docker $USER

# Verify installation
docker --version
docker compose version
```

### 3. Clone Your Repository
```bash
# Using HTTPS
git clone https://github.com/yourusername/TestPhotos.git
cd TestPhotos

# Or upload files directly via SCP
# scp -r /path/to/TestPhotos username@vps-ip:/home/username/
```

## 🔐 Environment Configuration

### 1. Set Up API Keys
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys
nano .env
```

Add your actual API keys:
```bash
# Historical Photos API Environment Variables
PINECONE_API_KEY=your_actual_pinecone_key_here
GEMINI_API_KEY=your_actual_gemini_key_here
GROQ_API_KEY=your_actual_groq_key_here

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=false
PORT=5000
```

### 2. Get API Keys
- **Pinecone:** https://www.pinecone.io/ (Free: 1M vectors)
- **Gemini:** https://makersuite.google.com/app/apikey (Free tier available)
- **Groq:** https://groq.com/ (Free tier available)

## 🚀 Deployment

### Quick Start
```bash
# Make deployment script executable
chmod +x deploy.sh

# Deploy application
./deploy.sh deploy
```

### Manual Steps
```bash
# 1. Build Docker image
docker-compose build

# 2. Start application
docker-compose up -d

# 3. Check status
docker-compose ps
```

## 🔍 Verification

### Health Check
```bash
# Basic health check
curl http://localhost:5000/health

# Expected response:
# {"status":"healthy","service":"historical-photos-api","version":"1.0.0"}
```

### API Status
```bash
# Check database connections and services
curl http://localhost:5000/api/status

# Expected response includes:
# - Pinecone connection status
# - Photo count in database
# - Service availability
```

### Test Search
```bash
# Test semantic search
curl 'http://localhost:5000/api/search/semantic?q=love%20is%20in%20the%20air&limit=5'

# Test with Gemini reranking (if enabled)
curl 'http://localhost:5000/api/search/semantic?q=war&limit=10&gemini=true'
```

## 🌐 Public Access Setup

### Option 1: Direct Port Access
```bash
# Allow port 5000 through firewall
sudo ufw allow 5000/tcp

# Access via: http://your-vps-ip:5000
```

### Option 2: Nginx Reverse Proxy (Recommended)
```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx configuration
sudo nano /etc/nginx/sites-available/historical-photos
```

Add this configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;  # or your VPS IP

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

```bash
# Enable site
sudo ln -s /etc/nginx/sites-available/historical-photos /etc/nginx/sites-enabled/

# Test and reload Nginx
sudo nginx -t
sudo systemctl reload nginx

# Allow HTTP through firewall
sudo ufw allow 'Nginx Full'
```

### Option 3: SSL with Let's Encrypt
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo systemctl enable certbot.timer
```

## 📊 Monitoring & Management

### View Logs
```bash
# Application logs
./deploy.sh logs

# Or manually:
docker-compose logs -f testphotos-api
```

### Application Status
```bash
# Check status
./deploy.sh status

# Container status
docker-compose ps

# System resources
docker stats
```

### Common Commands
```bash
# Restart application
./deploy.sh restart

# Stop application
./deploy.sh stop

# Update application
./deploy.sh update

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 🔧 Troubleshooting

### Application Won't Start
```bash
# Check logs for errors
docker-compose logs testphotos-api

# Common issues:
# 1. Missing API keys in .env
# 2. Insufficient memory (need 2GB+)
# 3. Port 5000 already in use
```

### API Endpoints Return 502
```bash
# Check if app is initializing
docker-compose logs -f testphotos-api

# Wait for model loading to complete
# Look for: "✅ Pinecone handler ready"
```

### Out of Memory Errors
```bash
# Check available memory
free -h

# Monitor Docker memory usage
docker stats

# Restart containers if needed
docker-compose restart
```

### Pinecone Connection Errors
```bash
# Verify API key is correct
grep PINECONE_API_KEY .env

# Test connection manually
docker-compose exec testphotos-api python3 -c "
import os
from pinecone import Pinecone
client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print([idx.name for idx in client.list_indexes()])
"
```

## 📈 Performance Optimization

### Resource Limits
Edit `docker-compose.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 4G      # Adjust based on your VPS
      cpus: '2.0'
    reservations:
      memory: 1G
      cpus: '0.5'
```

### Persistent Storage
```bash
# Create directories for persistent data
mkdir -p ./logs ./data

# Volumes are already configured in docker-compose.yml
```

### Backup Strategy
```bash
# Backup environment and logs
tar -czf backup-$(date +%Y%m%d).tar.gz .env logs/ main_app_outputs/

# Regular backups (crontab)
# 0 2 * * * cd /path/to/TestPhotos && tar -czf backup-$(date +\%Y\%m\%d).tar.gz .env logs/
```

## 🔄 Updates & Maintenance

### Update Application
```bash
# Pull latest code
git pull origin main

# Deploy updates
./deploy.sh update
```

### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d
```

### Clean Up
```bash
# Remove old Docker images
docker system prune -a

# Remove old logs
find logs/ -name "*.log" -mtime +30 -delete
```

## 🛡️ Security Best Practices

### Firewall Setup
```bash
# Enable UFW
sudo ufw enable

# Allow SSH
sudo ufw allow ssh

# Allow HTTP/HTTPS (if using Nginx)
sudo ufw allow 'Nginx Full'

# Or allow specific port
sudo ufw allow 5000/tcp
```

### Regular Updates
```bash
# Set up automatic security updates
sudo apt install unattended-upgrades -y
sudo dpkg-reconfigure -plow unattended-upgrades
```

### Environment Security
```bash
# Secure .env file permissions
chmod 600 .env

# Don't commit .env to git
echo ".env" >> .gitignore
```

## 📞 Support

### Useful Commands Summary
```bash
# Deploy fresh installation
./deploy.sh deploy

# Check if running
./deploy.sh status

# View logs
./deploy.sh logs

# Restart after issues
./deploy.sh restart

# Stop completely
./deploy.sh stop
```

### API Endpoints
- **Health Check:** `GET /health`
- **API Status:** `GET /api/status` 
- **Semantic Search:** `GET /api/search/semantic?q=query&limit=10`
- **Categories:** `GET /api/categories`

### Log Files
- Application logs: `docker-compose logs testphotos-api`
- Nginx logs: `/var/log/nginx/access.log`
- System logs: `/var/log/syslog`

---

## 🎉 Success!

Your Historical Photos API is now running on your VPS! The application provides:

✅ **Cloud-hosted vector search** with 858+ historical photos  
✅ **RESTful API** for semantic search and filtering  
✅ **AI-powered reranking** with Gemini (optional)  
✅ **Production-ready** with health checks and monitoring  
✅ **Scalable architecture** with Docker containers  

**Access your API at:** `http://your-vps-ip:5000` or your custom domain

Need help? Check the logs with `./deploy.sh logs` and refer to the troubleshooting section above.