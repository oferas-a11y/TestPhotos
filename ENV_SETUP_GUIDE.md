# 🔐 Environment Variables Setup Guide

Complete guide for setting up API keys and environment variables for your Historical Photos API deployment.

## 🎯 Where to Put Environment Variables

### **Option 1: .env File (Recommended)**

Create a `.env` file in your project root directory:

```bash
# On your VPS
cd /path/to/TestPhotos
nano .env
```

Add your API keys:
```bash
# Historical Photos API Environment Variables
# Required: Pinecone Vector Database
PINECONE_API_KEY=pc-your-actual-pinecone-key-here

# Optional: AI Enhancement Services  
GEMINI_API_KEY=your-actual-gemini-api-key-here
GROQ_API_KEY=your-actual-groq-api-key-here

# Flask Configuration (usually don't need to change)
FLASK_ENV=production
FLASK_DEBUG=false
PORT=5000
```

**Security:**
```bash
# Make sure only you can read the .env file
chmod 600 .env

# Verify it's not committed to git
echo ".env" >> .gitignore
```

### **Option 2: Docker Compose Override**

Create `docker-compose.override.yml`:
```yaml
version: '3.8'
services:
  testphotos-api:
    environment:
      - PINECONE_API_KEY=pc-your-actual-key-here
      - GEMINI_API_KEY=your-gemini-key-here
      - GROQ_API_KEY=your-groq-key-here
```

### **Option 3: System Environment Variables**

Export directly in your shell:
```bash
export PINECONE_API_KEY="pc-your-actual-key-here"
export GEMINI_API_KEY="your-gemini-key-here"  
export GROQ_API_KEY="your-groq-key-here"

# Make permanent (add to ~/.bashrc)
echo 'export PINECONE_API_KEY="pc-your-key-here"' >> ~/.bashrc
```

### **Option 4: Docker Run Command**

If running manually with docker:
```bash
docker run -d \
  -p 5000:5000 \
  -e PINECONE_API_KEY="pc-your-key-here" \
  -e GEMINI_API_KEY="your-gemini-key-here" \
  -e GROQ_API_KEY="your-groq-key-here" \
  historical-photos-api
```

## 🔑 How to Get API Keys

### **1. Pinecone (Required)**
- Website: https://www.pinecone.io/
- Plan: Free tier (1M vectors, perfect for your needs)
- Steps:
  1. Sign up for free account
  2. Create a new project  
  3. Go to API Keys section
  4. Copy your API key (starts with `pc-`)

**Example:** `PINECONE_API_KEY=pc-1234abcd-5678-90ef-ghij-klmnopqrstuv`

### **2. Gemini (Optional but Recommended)**
- Website: https://makersuite.google.com/app/apikey
- Plan: Free tier available
- Steps:
  1. Sign in with Google account
  2. Create new API key
  3. Copy the key

**Example:** `GEMINI_API_KEY=AIzaSyABC123def456GHI789jkl012MNO345pqr`

### **3. Groq (Optional)**
- Website: https://groq.com/
- Plan: Free tier available
- Steps:
  1. Sign up for account
  2. Go to API Keys section
  3. Generate new key

**Example:** `GROQ_API_KEY=gsk_abc123def456ghi789jkl012mno345pqr`

## 📋 Environment Variable Reference

### **Required Variables:**
| Variable | Purpose | Example | Required |
|----------|---------|---------|----------|
| `PINECONE_API_KEY` | Vector database access | `pc-1234...` | ✅ **YES** |

### **Optional Variables:**
| Variable | Purpose | Default | Required |
|----------|---------|---------|----------|
| `GEMINI_API_KEY` | AI reranking | None | ❌ Optional |
| `GROQ_API_KEY` | Alternative AI | None | ❌ Optional |
| `FLASK_ENV` | Flask environment | `production` | ❌ Optional |
| `FLASK_DEBUG` | Flask debug mode | `false` | ❌ Optional |
| `PORT` | Server port | `5000` | ❌ Optional |

## 🚀 Deployment Workflow

### **Complete Setup Process:**

1. **Clone repository on VPS:**
```bash
git clone https://github.com/yourusername/TestPhotos.git
cd TestPhotos
```

2. **Create .env file:**
```bash
cp .env.example .env  # If you have a template
# OR
nano .env            # Create from scratch
```

3. **Add your API keys:**
```bash
# Edit .env file with your actual keys
PINECONE_API_KEY=pc-your-actual-key-here
GEMINI_API_KEY=your-actual-key-here
GROQ_API_KEY=your-actual-key-here
```

4. **Deploy:**
```bash
chmod +x deploy.sh
./deploy.sh deploy
```

### **Verification:**

Check if environment variables are loaded:
```bash
# View logs to see startup
./deploy.sh logs

# Should see:
# ✅ PINECONE_API_KEY is configured  
# ⚠️ WARNING: GEMINI_API_KEY not set (if not provided)
```

Test API endpoints:
```bash
# Health check
curl http://localhost:5000/health

# API status (shows database connection)
curl http://localhost:5000/api/status
```

## 🔒 Security Best Practices

### **File Permissions:**
```bash
# Secure .env file
chmod 600 .env
chown $USER:$USER .env
```

### **Git Security:**
```bash
# Never commit API keys
echo ".env" >> .gitignore
echo "*.env" >> .gitignore

# Check if .env is ignored
git status  # Should not show .env file
```

### **VPS Security:**
```bash
# Use firewall
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # Or your chosen port

# Regular updates
sudo apt update && sudo apt upgrade -y
```

## 🔧 Troubleshooting

### **Common Issues:**

**❌ "PINECONE_API_KEY not set"**
- Check .env file exists: `ls -la .env`
- Check file contents: `cat .env`
- Check permissions: `ls -la .env` (should show `-rw-------`)

**❌ "Failed to connect to Pinecone"**
- Verify API key is correct (copy-paste from Pinecone dashboard)
- Check internet connection: `curl -I https://api.pinecone.io`
- Try regenerating API key in Pinecone dashboard

**❌ "Permission denied" on .env**
- Fix permissions: `chmod 600 .env`
- Check ownership: `chown $USER:$USER .env`

**❌ Docker can't read .env**
- Ensure .env is in same directory as docker-compose.yml
- Check docker-compose.yml references .env correctly

### **Debug Commands:**

```bash
# Check if variables are loaded in container
docker-compose exec testphotos-api env | grep -E "(PINECONE|GEMINI|GROQ)"

# Test Pinecone connection manually
docker-compose exec testphotos-api python3 -c "
import os
from pinecone import Pinecone
client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print([idx.name for idx in client.list_indexes()])
"
```

## 📝 Quick Reference

### **Essential Commands:**
```bash
# Create .env file
nano .env

# Deploy with environment
./deploy.sh deploy

# Check logs for environment loading
./deploy.sh logs

# Test API with environment
curl http://localhost:5000/api/status
```

### **File Locations:**
- **Environment file:** `/path/to/TestPhotos/.env`
- **Docker compose:** `/path/to/TestPhotos/docker-compose.yml`  
- **Application logs:** `docker-compose logs testphotos-api`

---

## ✅ Success Checklist

Before deploying, ensure you have:

- [ ] Pinecone API key (starts with `pc-`)
- [ ] .env file created with correct permissions (600)
- [ ] API keys added to .env file
- [ ] .env file in same directory as docker-compose.yml
- [ ] Firewall allows your chosen port
- [ ] Docker and Docker Compose installed

Your Historical Photos API is now ready for secure deployment on your 2GB VPS! 🚀