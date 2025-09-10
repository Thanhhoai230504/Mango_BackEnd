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
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import os
import gdown 

app = FastAPI(title="Mango Quality Checker API")

# ✅ Bật CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Link Google Drive (chia sẻ công khai)
DRIVE_URL = "https://drive.google.com/uc?id=1Huahb05L3-NGbFVIJGI_gBkqhBgOSirv"
MODEL_PATH = "best.pt"

# ✅ Tải model nếu chưa tồn tại
if not os.path.exists(MODEL_PATH):
    print("📥 Đang tải model từ Google Drive...")
    gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)

# ✅ Lazy-load YOLO model
model = None  

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)  # load khi lần đầu predict
    return model

# Thư mục lưu ảnh kết quả
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to the Mango Quality Checker API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ✅ Lấy model (lazy-load)
    model = get_model()
    results = model.predict(img, conf=0.5)

    boxes = results[0].boxes
    annotated_img = img.copy()
    response_data = []

    for box in boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        raw_label = model.names[cls_id]

        # ✅ Chuẩn hóa nhãn
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

        # Vẽ khung + text
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated_img,
            f"{label} {conf:.2f} {emoji}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        response_data.append({
            "label": label,
            "confidence": round(conf * 100, 2),
            "emoji": emoji,
            "message": message
        })

    # Lưu ảnh đã gắn khung
    output_filename = f"{uuid.uuid4().hex}.jpg"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, annotated_img)

    return {
        "results": response_data,
        "image_url": f"/download/{output_filename}"
    }

@app.get("/download/{filename}")
def download_file(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg", filename=filename)
    return JSONResponse(content={"error": "File not found"}, status_code=404)
##############################################




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
        # )
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