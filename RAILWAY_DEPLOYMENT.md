# 🚂 Railway Deployment Guide - Historical Photos API

Deploy your Historical Photos API on Railway with 8GB RAM and 8 vCPU for optimal performance at just $5/month minimum.

## 🎯 Why Railway?

### **Superior Performance**
- **8GB RAM** - 6x more than typical VPS options
- **8 vCPU** - Parallel processing for fast responses
- **Auto-scaling** - Handle traffic spikes automatically
- **$5 minimum** - Pay only for what you use

### **Developer Experience**
- **GitHub integration** - Push to deploy
- **Built-in monitoring** - Logs and metrics
- **Custom domains** - Free SSL certificates
- **Zero configuration** - Just connect and deploy

## 📋 Prerequisites

### **Required Accounts**
- **Railway Account**: Sign up at https://railway.app
- **GitHub Repository**: Your TestPhotos repository
- **API Keys**: Pinecone (required), Gemini (recommended), Groq (optional)

### **API Key Requirements**
- **Pinecone**: Vector database - https://www.pinecone.io/ (Free: 1M vectors)
- **Gemini**: AI reranking - https://makersuite.google.com/app/apikey (Free tier)
- **Groq**: Advanced features - https://groq.com/ (Free tier)

## 🚀 Deployment Steps

### **1. Connect GitHub Repository**

1. **Login to Railway**:
   - Go to https://railway.app
   - Sign in with GitHub account

2. **Create New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your TestPhotos repository

3. **Railway Auto-Detection**:
   - Railway will detect the `railway.json` file
   - It will use `Dockerfile.railway` automatically
   - Build process starts immediately

### **2. Configure Environment Variables**

1. **Access Variables**:
   - Go to your Railway project dashboard  
   - Click on the "Variables" tab
   - Click "Add Variable"

2. **Required Variables**:
```
PINECONE_API_KEY = your_actual_pinecone_api_key_here
```

3. **Optional Variables** (for enhanced features):
```
GEMINI_API_KEY = your_actual_gemini_api_key_here
GROQ_API_KEY = your_actual_groq_api_key_here
```

4. **Railway Auto-Variables** (automatically set):
   - `PORT` - Railway assigns this automatically
   - `RAILWAY_ENVIRONMENT` - Environment name
   - Other Railway-specific variables

### **3. Deploy and Monitor**

1. **Deployment Process**:
   - Railway builds your Docker image
   - Deploys to cloud infrastructure
   - Assigns a public URL (e.g., `yourapp.railway.app`)

2. **Monitor Deployment**:
   - Watch build logs in Railway dashboard
   - Check deployment status
   - View application logs

## 🔍 Verification

### **Health Checks**
```bash
# Basic health check
curl https://yourapp.railway.app/health

# Expected response:
# {"status":"healthy","service":"historical-photos-api","version":"1.0.0"}
```

### **API Status**
```bash
# Check database connections and services
curl https://yourapp.railway.app/api/status

# Expected response includes:
# - Pinecone connection status
# - Photo count in database (858+)
# - Service availability
```

### **Test Search**
```bash
# Test semantic search
curl 'https://yourapp.railway.app/api/search/semantic?q=love%20is%20in%20the%20air&limit=5'

# Test with Gemini reranking (if configured)
curl 'https://yourapp.railway.app/api/search/semantic?q=historical%20photos&limit=10&gemini=true'
```

## 🌐 Custom Domain Setup

### **Add Custom Domain**
1. **In Railway Dashboard**:
   - Go to "Settings" tab
   - Click "Domains"
   - Add your custom domain

2. **DNS Configuration**:
   - Add CNAME record: `your-domain.com` → `yourapp.railway.app`
   - Railway provides free SSL certificates

3. **Verify Domain**:
   - Railway validates domain ownership
   - SSL certificate issued automatically

## 💰 Cost Estimation

### **Railway Pricing**
- **Base**: $5 minimum monthly usage
- **RAM**: $10 per GB/month
- **CPU**: $20 per vCPU/month
- **Egress**: $0.05 per GB/month

### **Your App Usage** (estimated):
- **Memory**: ~1GB average usage = $10/month
- **CPU**: ~0.5 vCPU average = $10/month
- **Egress**: ~5GB/month = $0.25/month
- **Total**: ~$20/month for moderate usage
- **Scales down**: To $5/month during low usage

### **Cost vs Performance**:
- **Railway 8GB**: ~$20/month for excellent performance
- **VPS 2GB**: ~$6/month but limited resources
- **Railway wins**: Better performance, managed platform, auto-scaling

## 📊 Performance Expectations

### **With 8GB RAM + 8 vCPU**:
- **Memory usage**: 15% of available (very comfortable)
- **Response times**: 100-300ms per search
- **Concurrent users**: 20-50+ simultaneous searches
- **Model loading**: Ultra-fast initialization
- **Gemini reranking**: Zero performance impact

### **API Throughput**:
- **Searches**: 200+ requests/minute
- **Embeddings**: Batch processing capable
- **Scaling**: Automatic based on demand

## 🔧 Management & Monitoring

### **View Logs**
```bash
# In Railway dashboard:
# 1. Go to "Deployments" tab
# 2. Click on latest deployment
# 3. View real-time logs
```

### **Key Metrics to Monitor**:
- **Memory usage** (should stay under 2GB)
- **CPU usage** (will spike during searches)
- **Response times** (target <500ms)
- **Error rates** (should be <1%)

### **Common Commands**:
```bash
# Health check
curl https://yourapp.railway.app/health

# Get service status
curl https://yourapp.railway.app/api/status

# Search test
curl 'https://yourapp.railway.app/api/search/semantic?q=test&limit=3'
```

## 🔄 Updates & CI/CD

### **Automatic Deployments**:
1. **Push to GitHub**: 
   ```bash
   git push origin main
   ```

2. **Railway Auto-Deploy**:
   - Detects GitHub push
   - Builds new Docker image
   - Deploys automatically
   - Zero downtime deployment

### **Manual Deployments**:
- Use Railway dashboard
- Trigger deployment manually
- Roll back to previous versions

## 🛠️ Troubleshooting

### **Common Issues**

**❌ "Build Failed"**
- Check `Dockerfile.railway` exists
- Verify `requirements-railway.txt` is valid
- Review build logs in Railway dashboard

**❌ "Health Check Failed"**
- Ensure app starts on correct PORT (Railway sets this)
- Check environment variables are set
- Verify `/health` endpoint responds

**❌ "API Endpoints Return 500"**
- Check Pinecone API key is correct
- Verify model loading in logs
- Monitor memory usage (shouldn't exceed 6GB)

**❌ "Slow Performance"**
- Check if using `requirements-railway.txt` (full PyTorch)
- Monitor CPU usage during searches
- Verify sentence-transformers model loading

### **Debug Commands**:
```bash
# Check environment variables (in Railway dashboard logs)
# Look for startup messages:
# ✅ PINECONE_API_KEY is configured
# ✅ Pinecone connected successfully!

# Monitor resource usage
# Railway dashboard shows real-time metrics

# Test individual components
curl https://yourapp.railway.app/api/categories
```

### **Performance Optimization**:
```bash
# If memory usage is high:
# 1. Check for memory leaks in logs
# 2. Monitor model loading efficiency
# 3. Verify garbage collection

# If response times are slow:
# 1. Check Pinecone connection latency
# 2. Monitor Gemini API response times
# 3. Verify sentence-transformers performance
```

## 📈 Scaling & Production

### **Auto-Scaling Features**:
- Railway automatically scales based on demand
- CPU and memory adjust dynamically
- Pay only for actual usage

### **Production Checklist**:
- [ ] Environment variables configured
- [ ] Custom domain setup (optional)
- [ ] Monitoring alerts configured
- [ ] Backup strategy for environment config
- [ ] Documentation updated with Railway URLs

### **Best Practices**:
- **Monitor costs** regularly in Railway dashboard
- **Set spending alerts** to avoid surprises
- **Use staging environment** for testing changes
- **Keep API keys secure** (never commit to git)

## 📞 Support & Resources

### **Railway Resources**:
- **Documentation**: https://docs.railway.app/
- **Discord**: Railway community support
- **Status**: https://status.railway.app/

### **API Endpoints** (replace with your Railway URL):
- **Health**: `https://yourapp.railway.app/health`
- **Status**: `https://yourapp.railway.app/api/status`
- **Search**: `https://yourapp.railway.app/api/search/semantic?q=query`
- **Categories**: `https://yourapp.railway.app/api/categories`

### **Project Files**:
- **Railway Config**: `railway.json`
- **Dockerfile**: `Dockerfile.railway`
- **Requirements**: `requirements-railway.txt`
- **Entrypoint**: `railway-entrypoint.sh`

---

## 🎉 Success!

Your Historical Photos API is now running on Railway with:

✅ **8GB RAM** - Plenty of headroom for growth  
✅ **8 vCPU** - Fast, responsive searches  
✅ **Auto-scaling** - Handles traffic spikes  
✅ **$5 minimum** - Cost-effective and predictable  
✅ **858+ photos** - Full semantic search capability  
✅ **Gemini AI** - Enhanced search reranking  
✅ **Production-ready** - Monitoring and SSL included  

**Access your API**: `https://yourapp.railway.app`

Railway provides the perfect balance of performance, developer experience, and cost for your Historical Photos API! 🚂