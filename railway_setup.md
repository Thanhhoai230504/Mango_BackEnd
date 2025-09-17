# 🚀 Railway Deployment Guide (Simplified)

## Step 1: Prepare Files
Upload these 2 files to Railway:
- `main.py` (the simplified backend)
- `requirements.txt` (minimal dependencies)

## Step 2: Railway Settings
**Build Command:** (leave empty - Railway auto-detects)
**Start Command:** 
```
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

## Step 3: Environment Variables
Railway automatically sets `PORT` - no need to add anything.

## Step 4: Deploy Process
1. Connect your GitHub repo to Railway
2. Railway will automatically build and deploy
3. First build takes 5-10 minutes (downloading ML dependencies)
4. Model download takes another 2-3 minutes on first run

## Step 5: Update Frontend
Update your `.env` file:
```
VITE_API_BASE_URL=https://your-project-name.up.railway.app
```

## Expected Timeline:
- **Build**: 5-10 minutes
- **First Request**: 2-3 minutes (model download)
- **Subsequent Requests**: 3-7 seconds

## Troubleshooting:
1. **Build fails**: Check Build Logs tab
2. **Memory issues**: Upgrade to Railway Pro ($5/month)
3. **Timeout**: First request may take 3+ minutes

## Alternative: No Docker Approach
If still having issues, try:
1. Remove any Dockerfile
2. Use only `main.py` and `requirements.txt`
3. Let Railway use Nixpacks (automatic detection)

## Pro Tips:
- Railway Pro gives 8GB RAM vs 512MB free
- Pro also removes sleep mode
- Very worth it for ML apps ($5/month)