# Railway Deployment Setup

## 🚀 Quick Deploy Steps

### 1. Prepare Files

Upload these files to your Railway project:

- `main.py` → rename to `main.py`
- `requirements.txt` → rename to `requirements.txt`
- `Dockerfile` → rename to `Dockerfile`

### 2. Railway Configuration

**Build Settings:**

- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`

**Environment Variables:**

- `PORT` (automatically set by Railway)
- `PYTHONUNBUFFERED=1`
- `PYTHONDONTWRITEBYTECODE=1`

### 3. Resource Optimization

**Memory Usage:**

- Reduced image processing size to 640x480
- Single worker process
- Aggressive garbage collection
- Limited file storage (10 files max)

**Docker Optimization:**

- Multi-stage build to reduce image size
- Headless OpenCV (no GUI dependencies)
- Minimal system dependencies
- Clean package cache

### 4. Expected Performance

**First Deploy:**

- Build time: 5-8 minutes
- Model download: 2-3 minutes
- Total cold start: ~10 minutes

**Runtime:**

- Warm requests: 3-7 seconds
- Memory usage: ~800MB-1.2GB
- Image size: ~2.5GB (vs 7.7GB before)

### 5. Monitoring

Check these endpoints after deployment:

- `GET /` - Basic health check
- `GET /health` - Detailed status with memory usage
- `GET /stats` - System statistics

### 6. Troubleshooting

**If build still fails:**

1. Try without Dockerfile first (use requirements.txt only)
2. Check Railway logs for specific error
3. Reduce dependencies further if needed

**Memory issues:**

- Railway free tier has 512MB RAM limit
- Upgrade to Pro ($5/month) for 8GB RAM
- Monitor `/stats` endpoint for memory usage

### 7. Alternative: Nixpacks Build

If Dockerfile fails, try Nixpacks (Railway's default):

1. Remove Dockerfile
2. Keep only `main.py` and `requirements.txt`
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

This will use Railway's automatic build detection.
