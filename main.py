# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import uuid
# import os
# import gdown 

# app = FastAPI(title="Mango Quality Checker API")

# # ✅ Bật CORS cho frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# # ✅ Link Google Drive (chia sẻ công khai)
# DRIVE_URL = "https://drive.google.com/uc?id=1Huahb05L3-NGbFVIJGI_gBkqhBgOSirv"
# MODEL_PATH = "best.pt"

# # ✅ Tải model nếu chưa tồn tại
# if not os.path.exists(MODEL_PATH):
#     print("📥 Đang tải model từ Google Drive...")
#     gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# # Load YOLO model
# model = YOLO(MODEL_PATH)

# # Thư mục lưu ảnh kết quả
# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Mango Quality Checker API!"}

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Đọc ảnh từ upload
#     contents = await file.read()
#     nparr = np.frombuffer(contents, np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

#     # Chạy dự đoán với YOLO
#     results = model.predict(img, conf=0.5)

#     boxes = results[0].boxes
#     annotated_img = img.copy()
#     response_data = []

#     for box in boxes:
#         cls_id = int(box.cls[0].item())
#         conf = float(box.conf[0].item())
#         raw_label = model.names[cls_id]

#         # ✅ Chuẩn hóa nhãn
#         if "fresh" in raw_label.lower():
#             label = "fresh"
#             color = (0, 255, 0)
#             emoji = "✅🍋"
#             message = "Xoài ngon rồi đấy"
#         else:
#             label = "rotten"
#             color = (0, 0, 255)
#             emoji = "❌🟤"
#             message = "Ui, xoài hỏng rồi"

#         # Vẽ khung + text
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
#         cv2.putText(
#             annotated_img,
#             f"{label} {conf:.2f} {emoji}",
#             (x1, y1 - 10),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.8,
#             color,
#             2
#         )

#         response_data.append({
#             "label": label,
#             "confidence": round(conf * 100, 2),
#             "emoji": emoji,
#             "message": message
#         })

#     # Lưu ảnh đã gắn khung
#     output_filename = f"{uuid.uuid4().hex}.jpg"
#     output_path = os.path.join(OUTPUT_DIR, output_filename)
#     cv2.imwrite(output_path, annotated_img)

#     return {
#         "results": response_data,
#         "image_url": f"/download/{output_filename}"
#     }


# @app.get("/download/{filename}")
# def download_file(filename: str):
#     file_path = os.path.join(OUTPUT_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(file_path, media_type="image/jpeg", filename=filename)
#     return JSONResponse(content={"error": "File not found"}, status_code=404)

#########################################

# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import uuid
# import os
# import gdown 
# import asyncio
# from threading import Timer
# import requests
# from PIL import Image
# import io

# app = FastAPI(title="Mango Quality Checker API")

# # ✅ Bật CORS cho frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ✅ Link Google Drive (chia sẻ công khai)
# DRIVE_URL = "https://drive.google.com/uc?id=1Huahb05L3-NGbFVIJGI_gBkqhBgOSirv"
# MODEL_PATH = "best.pt"

# # Global model variable to avoid reloading
# model = None

# # ✅ Keep-alive mechanism
# def keep_alive():
#     try:
#         # Thay thế bằng URL của bạn trên Render
#         requests.get("https://mango-backend-2htc.onrender.com/health", timeout=30)
#         print("✅ Keep-alive ping sent")
#     except Exception as e:
#         print(f"❌ Keep-alive failed: {e}")
    
#     # Schedule next ping in 10 minutes
#     Timer(600, keep_alive).start()

# # ✅ Load model một lần duy nhất
# async def load_model():
#     global model
#     if model is None:
#         # Tải model nếu chưa tồn tại
#         if not os.path.exists(MODEL_PATH):
#             print("📥 Đang tải model từ Google Drive...")
#             gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        
#         print("🔄 Loading YOLO model...")
#         model = YOLO(MODEL_PATH)
#         print("✅ Model loaded successfully!")

# # Thư mục lưu ảnh kết quả
# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# @app.on_event("startup")
# async def startup_event():
#     # Load model khi khởi động
#     await load_model()
#     # Bắt đầu keep-alive mechanism
#     Timer(600, keep_alive).start()  # 10 minutes
#     print("🚀 Server started with optimizations!")

# @app.get("/")
# async def root():
#     return {"message": "Welcome to the Mango Quality Checker API!", "status": "ready"}

# @app.get("/health")
# async def health_check():
#     """Health check endpoint for keep-alive"""
#     return {"status": "healthy", "model_loaded": model is not None}

# # ✅ Tối ưu hóa xử lý ảnh
# def optimize_image(img, max_size=1024):
#     """Resize ảnh để giảm thời gian xử lý"""
#     height, width = img.shape[:2]
#     if max(height, width) > max_size:
#         if width > height:
#             new_width = max_size
#             new_height = int((height * max_size) / width)
#         else:
#             new_height = max_size
#             new_width = int((width * max_size) / height)
        
#         img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
#     return img

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     global model
    
#     try:
#         # Đảm bảo model đã được load
#         if model is None:
#             await load_model()
        
#         # ✅ Optimize file reading
#         contents = await file.read()
        
#         # ✅ Sử dụng PIL để đọc ảnh nhanh hơn
#         image_pil = Image.open(io.BytesIO(contents))
#         img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
#         # ✅ Tối ưu kích thước ảnh
#         img = optimize_image(img, max_size=800)

#         # ✅ Chạy dự đoán với confidence thấp hơn để nhanh hơn
#         results = model.predict(img, conf=0.3, verbose=False)

#         boxes = results[0].boxes
#         annotated_img = img.copy()
#         response_data = []

#         if boxes is not None and len(boxes) > 0:
#             for box in boxes:
#                 cls_id = int(box.cls[0].item())
#                 conf = float(box.conf[0].item())
#                 raw_label = model.names[cls_id]

#                 # ✅ Chuẩn hóa nhãn
#                 if "fresh" in raw_label.lower():
#                     label = "fresh"
#                     color = (0, 255, 0)
#                     emoji = "✅🍋"
#                     message = "Xoài ngon rồi đấy"
#                 else:
#                     label = "rotten"
#                     color = (0, 0, 255)
#                     emoji = "❌🟤"
#                     message = "Ui, xoài hỏng rồi"

#                 # Vẽ khung + text
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
#                 # ✅ Font size nhỏ hơn để vẽ nhanh hơn
#                 cv2.putText(
#                     annotated_img,
#                     f"{label} {conf:.1f}",
#                     (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     color,
#                     1
#                 )

#                 response_data.append({
#                     "label": label,
#                     "confidence": round(conf * 100, 1),  # Làm tròn ít hơn
#                     "emoji": emoji,
#                     "message": message
#                 })

#         # ✅ Lưu ảnh với chất lượng thấp hơn để nhanh hơn
#         output_filename = f"{uuid.uuid4().hex}.jpg"
#         output_path = os.path.join(OUTPUT_DIR, output_filename)
        
#         # Compress image
#         cv2.imwrite(output_path, annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 80])

#         return {
#             "results": response_data,
#             "image_url": f"/download/{output_filename}",
#             "total_detected": len(response_data)
#         }
        
#     except Exception as e:
#         print(f"❌ Prediction error: {e}")
#         return JSONResponse(
#             content={"error": f"Prediction failed: {str(e)}"}, 
#             status_code=500
#         )

# @app.get("/download/{filename}")
# def download_file(filename: str):
#     file_path = os.path.join(OUTPUT_DIR, filename)
#     if os.path.exists(file_path):
#         return FileResponse(
#             file_path, 
#             media_type="image/jpeg", 
#             filename=filename,
#             headers={"Cache-Control": "public, max-age=3600"}  # Cache 1 hour
#         )
#     return JSONResponse(content={"error": "File not found"}, status_code=404)

# # ✅ Endpoint warm-up
# @app.get("/warmup")
# async def warmup():
#     """Endpoint để làm nóng server"""
#     if model is None:
#         await load_model()
#     return {"status": "warmed up", "message": "Server is ready!"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


############################################


# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse, FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import uuid
# import os
# import gdown
# import asyncio
# import threading
# from concurrent.futures import ThreadPoolExecutor
# import logging
# from contextlib import asynccontextmanager
# import time

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Global variables
# model = None
# model_loaded = False
# executor = ThreadPoolExecutor(max_workers=2)

# # Configuration
# DRIVE_URL = "https://drive.google.com/uc?id=1Huahb05L3-NGbFVIJGI_gBkqhBgOSirv"
# MODEL_PATH = "best.pt"
# OUTPUT_DIR = "outputs"
# MAX_IMAGE_SIZE = (800, 600)  # Reduced size for faster processing
# CLEANUP_INTERVAL = 3600  # Clean up files every hour

# async def load_model():
#     """Load YOLO model asynchronously"""
#     global model, model_loaded
    
#     try:
#         logger.info("Starting model loading...")
        
#         # Download model if not exists
#         if not os.path.exists(MODEL_PATH):
#             logger.info("📥 Downloading model from Google Drive...")
#             loop = asyncio.get_event_loop()
#             await loop.run_in_executor(
#                 executor, 
#                 gdown.download, 
#                 DRIVE_URL, 
#                 MODEL_PATH, 
#                 False
#             )
#             logger.info("✅ Model downloaded successfully")
        
#         # Load model in thread pool to avoid blocking
#         logger.info("🔄 Loading YOLO model...")
#         loop = asyncio.get_event_loop()
#         model = await loop.run_in_executor(executor, YOLO, MODEL_PATH)
        
#         # Warm up model with dummy prediction
#         logger.info("🔥 Warming up model...")
#         dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
#         await loop.run_in_executor(executor, model.predict, dummy_img, 0.5)
        
#         model_loaded = True
#         logger.info("✅ Model loaded and warmed up successfully!")
        
#     except Exception as e:
#         logger.error(f"❌ Error loading model: {str(e)}")
#         model_loaded = False

# def cleanup_old_files():
#     """Clean up old generated files"""
#     try:
#         if os.path.exists(OUTPUT_DIR):
#             now = time.time()
#             for filename in os.listdir(OUTPUT_DIR):
#                 file_path = os.path.join(OUTPUT_DIR, filename)
#                 if os.path.isfile(file_path):
#                     # Delete files older than 1 hour
#                     if now - os.path.getctime(file_path) > 3600:
#                         os.remove(file_path)
#                         logger.info(f"Cleaned up old file: {filename}")
#     except Exception as e:
#         logger.error(f"Error during cleanup: {str(e)}")

# async def periodic_cleanup():
#     """Run cleanup periodically"""
#     while True:
#         await asyncio.sleep(CLEANUP_INTERVAL)
#         cleanup_old_files()

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("🚀 Starting FastAPI application...")
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     # Start model loading in background
#     asyncio.create_task(load_model())
    
#     # Start periodic cleanup
#     asyncio.create_task(periodic_cleanup())
    
#     yield
    
#     # Shutdown
#     logger.info("⏹️ Shutting down application...")
#     executor.shutdown(wait=True)

# app = FastAPI(
#     title="Optimized Mango Quality Checker API",
#     description="Fast ML-powered mango quality detection",
#     version="2.0.0",
#     lifespan=lifespan
# )

# # CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def resize_image(image: np.ndarray, max_size: tuple = MAX_IMAGE_SIZE) -> np.ndarray:
#     """Resize image while maintaining aspect ratio"""
#     height, width = image.shape[:2]
#     max_width, max_height = max_size
    
#     # Calculate scaling factor
#     scale = min(max_width / width, max_height / height)
    
#     if scale < 1:
#         new_width = int(width * scale)
#         new_height = int(height * scale)
#         return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
#     return image

# def process_image_sync(img: np.ndarray) -> tuple:
#     """Process image synchronously in thread pool"""
#     try:
#         # Resize for faster processing
#         img_resized = resize_image(img)
        
#         # Run prediction
#         results = model.predict(img_resized, conf=0.5, verbose=False)
#         boxes = results[0].boxes
        
#         # Create annotated image
#         annotated_img = img_resized.copy()
#         response_data = []

#         for box in boxes:
#             cls_id = int(box.cls[0].item())
#             conf = float(box.conf[0].item())
#             raw_label = model.names[cls_id]

#             # Normalize labels
#             if "fresh" in raw_label.lower():
#                 label = "fresh"
#                 color = (0, 255, 0)
#                 emoji = "✅🍋"
#                 message = "Xoài ngon rồi đấy"
#             else:
#                 label = "rotten"
#                 color = (0, 0, 255)
#                 emoji = "❌🟤"
#                 message = "Ui, xoài hỏng rồi"

#             # Draw bounding box and text
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#             cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
#             cv2.putText(
#                 annotated_img,
#                 f"{label} {conf:.2f} {emoji}",
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.8,
#                 color,
#                 2
#             )

#             response_data.append({
#                 "label": label,
#                 "confidence": round(conf * 100, 2),
#                 "emoji": emoji,
#                 "message": message
#             })

#         return annotated_img, response_data
        
#     except Exception as e:
#         logger.error(f"Error in image processing: {str(e)}")
#         raise e

# @app.get("/")
# async def root():
#     """Health check endpoint"""
#     return {
#         "message": "Optimized Mango Quality Checker API is running!",
#         "model_loaded": model_loaded,
#         "version": "2.0.0"
#     }

# @app.get("/health")
# async def health_check():
#     """Detailed health check"""
#     return {
#         "status": "healthy" if model_loaded else "loading",
#         "model_loaded": model_loaded,
#         "timestamp": time.time()
#     }

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     """Predict mango quality from uploaded image"""
    
#     # Check if model is loaded
#     if not model_loaded:
#         raise HTTPException(
#             status_code=503, 
#             detail="Model is still loading. Please wait a moment and try again."
#         )
    
#     # Validate file
#     if not file.content_type.startswith('image/'):
#         raise HTTPException(status_code=400, detail="File must be an image")
    
#     try:
#         # Read and decode image
#         logger.info(f"Processing image: {file.filename}")
#         contents = await file.read()
        
#         if len(contents) == 0:
#             raise HTTPException(status_code=400, detail="Empty file uploaded")
        
#         nparr = np.frombuffer(contents, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         if img is None:
#             raise HTTPException(status_code=400, detail="Could not decode image")
        
#         # Process image in thread pool
#         loop = asyncio.get_event_loop()
#         annotated_img, response_data = await loop.run_in_executor(
#             executor,
#             process_image_sync,
#             img
#         )
        
#         # Save annotated image
#         output_filename = f"{uuid.uuid4().hex}.jpg"
#         output_path = os.path.join(OUTPUT_DIR, output_filename)
        
#         # Save in thread pool to avoid blocking
#         await loop.run_in_executor(
#             executor,
#             cv2.imwrite,
#             output_path,
#             annotated_img
#         )
        
#         logger.info(f"Successfully processed image with {len(response_data)} detections")
        
#         return {
#             "results": response_data,
#             "image_url": f"/download/{output_filename}",
#             "processing_time": "optimized"
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.get("/download/{filename}")
# async def download_file(filename: str):
#     """Download processed image"""
#     # Sanitize filename to prevent directory traversal
#     filename = os.path.basename(filename)
#     file_path = os.path.join(OUTPUT_DIR, filename)
    
#     if os.path.exists(file_path):
#         return FileResponse(
#             file_path, 
#             media_type="image/jpeg", 
#             filename=filename,
#             headers={"Cache-Control": "max-age=3600"}  # Cache for 1 hour
#         )
    
#     raise HTTPException(status_code=404, detail="File not found")

# @app.delete("/cleanup")
# async def manual_cleanup():
#     """Manual cleanup endpoint for old files"""
#     try:
#         cleanup_old_files()
#         return {"message": "Cleanup completed successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0", 
#         port=8000, 
#         reload=False,  # Disable reload in production
#         access_log=False  # Disable access logs for better performance
#     )


######################################
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import os
import gdown
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from contextlib import asynccontextmanager
import time
import gc
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_loaded = False
executor = ThreadPoolExecutor(max_workers=1)  # Reduced for Railway

# Configuration optimized for Railway
DRIVE_URL = "https://drive.google.com/uc?id=1Huahb05L3-NGbFVIJGI_gBkqhBgOSirv"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "outputs"
MAX_IMAGE_SIZE = (640, 480)  # Smaller for Railway
CLEANUP_INTERVAL = 1800  # Clean up every 30 minutes
MAX_FILES = 10  # Keep only 10 recent files

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

async def load_model():
    """Load YOLO model with memory optimization"""
    global model, model_loaded
    
    try:
        logger.info(f"Starting model loading... Memory: {get_memory_usage():.1f}MB")
        
        # Download model if not exists
        if not os.path.exists(MODEL_PATH):
            logger.info("📥 Downloading model from Google Drive...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor, 
                gdown.download, 
                DRIVE_URL, 
                MODEL_PATH, 
                False
            )
            logger.info("✅ Model downloaded successfully")
        
        # Load model with memory optimization
        logger.info("🔄 Loading YOLO model...")
        loop = asyncio.get_event_loop()
        
        def load_yolo():
            # Force garbage collection before loading
            gc.collect()
            return YOLO(MODEL_PATH)
        
        model = await loop.run_in_executor(executor, load_yolo)
        
        # Warm up with smaller dummy image
        logger.info("🔥 Warming up model...")
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        await loop.run_in_executor(executor, model.predict, dummy_img, 0.5)
        
        # Force garbage collection after warmup
        gc.collect()
        
        model_loaded = True
        logger.info(f"✅ Model loaded successfully! Memory: {get_memory_usage():.1f}MB")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        model_loaded = False

def cleanup_old_files():
    """Clean up old files with size limit"""
    try:
        if os.path.exists(OUTPUT_DIR):
            files = []
            for filename in os.listdir(OUTPUT_DIR):
                file_path = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(file_path):
                    files.append((file_path, os.path.getctime(file_path)))
            
            # Sort by creation time (newest first)
            files.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only MAX_FILES recent files
            for file_path, _ in files[MAX_FILES:]:
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up old file: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Error removing file {file_path}: {e}")
            
            # Force garbage collection after cleanup
            gc.collect()
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

async def periodic_cleanup():
    """Run cleanup periodically"""
    while True:
        await asyncio.sleep(CLEANUP_INTERVAL)
        cleanup_old_files()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Railway FastAPI application...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Start model loading in background
    asyncio.create_task(load_model())
    
    # Start periodic cleanup
    asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    logger.info("⏹️ Shutting down application...")
    executor.shutdown(wait=True)

app = FastAPI(
    title="Railway Mango Quality Checker API",
    description="Optimized ML-powered mango quality detection for Railway",
    version="2.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def resize_image(image: np.ndarray, max_size: tuple = MAX_IMAGE_SIZE) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    height, width = image.shape[:2]
    max_width, max_height = max_size
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def process_image_sync(img: np.ndarray) -> tuple:
    """Process image synchronously with memory optimization"""
    try:
        # Force garbage collection before processing
        gc.collect()
        
        # Resize for faster processing and less memory usage
        img_resized = resize_image(img)
        
        # Run prediction with lower confidence for faster processing
        results = model.predict(img_resized, conf=0.4, verbose=False, imgsz=320)
        boxes = results[0].boxes
        
        # Create annotated image
        annotated_img = img_resized.copy()
        response_data = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_label = model.names[cls_id]

            # Normalize labels
            if "fresh" in raw_label.lower():
                label = "fresh"
                color = (0, 255, 0)
                emoji = "✅🍋"
                message = "Xoài ngon rồi đấy"
            else:
                label = "rotten"
                color = (0, 0, 255)
                emoji = "❌🟤"
                message = "Ui, xoài hỏng rồi"

            # Draw bounding box and text
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated_img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            response_data.append({
                "label": label,
                "confidence": round(conf * 100, 2),
                "emoji": emoji,
                "message": message
            })

        # Force garbage collection after processing
        gc.collect()
        
        return annotated_img, response_data
        
    except Exception as e:
        logger.error(f"Error in image processing: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    memory_mb = get_memory_usage()
    return {
        "message": "Railway Mango Quality Checker API is running!",
        "model_loaded": model_loaded,
        "version": "2.1.0",
        "memory_usage_mb": round(memory_mb, 1),
        "platform": "Railway"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    memory_mb = get_memory_usage()
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "memory_usage_mb": round(memory_mb, 1),
        "timestamp": time.time()
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict mango quality from uploaded image"""
    
    # Check if model is loaded
    if not model_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model is still loading. Please wait a moment and try again."
        )
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (limit to 10MB for Railway)
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum 10MB allowed.")
    
    try:
        # Read and decode image
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()
        
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Check memory before processing
        memory_before = get_memory_usage()
        logger.info(f"Memory before processing: {memory_before:.1f}MB")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Process image in thread pool
        loop = asyncio.get_event_loop()
        annotated_img, response_data = await loop.run_in_executor(
            executor,
            process_image_sync,
            img
        )
        
        # Save annotated image with compression
        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save with compression to reduce file size
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        await loop.run_in_executor(
            executor,
            cv2.imwrite,
            output_path,
            annotated_img,
            encode_params
        )
        
        # Clean up memory
        del img, annotated_img, nparr, contents
        gc.collect()
        
        memory_after = get_memory_usage()
        logger.info(f"Memory after processing: {memory_after:.1f}MB")
        logger.info(f"Successfully processed image with {len(response_data)} detections")
        
        return {
            "results": response_data,
            "image_url": f"/download/{output_filename}",
            "processing_time": "optimized",
            "memory_usage_mb": round(memory_after, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        # Clean up memory on error
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed image"""
    # Sanitize filename to prevent directory traversal
    filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path, 
            media_type="image/jpeg", 
            filename=filename,
            headers={"Cache-Control": "max-age=1800"}  # Cache for 30 minutes
        )
    
    raise HTTPException(status_code=404, detail="File not found")

@app.delete("/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint for old files"""
    try:
        cleanup_old_files()
        memory_mb = get_memory_usage()
        return {
            "message": "Cleanup completed successfully",
            "memory_usage_mb": round(memory_mb, 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        memory_mb = get_memory_usage()
        file_count = len(os.listdir(OUTPUT_DIR)) if os.path.exists(OUTPUT_DIR) else 0
        
        return {
            "memory_usage_mb": round(memory_mb, 1),
            "model_loaded": model_loaded,
            "output_files": file_count,
            "max_files": MAX_FILES,
            "max_image_size": MAX_IMAGE_SIZE
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "railway_main:app", 
        host="0.0.0.0", 
        port=port, 
        reload=False,
        access_log=False
    )