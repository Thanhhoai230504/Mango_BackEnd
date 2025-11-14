
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
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
import base64
from typing import List, Dict
import io
from datetime import datetime
import google.generativeai as genai
from pydantic import BaseModel
GEMINI_API_KEY = "AIzaSyBKG3_rNmBwGKvBhgHazDptcPqg77dEWFk" 
genai.configure(api_key=GEMINI_API_KEY)


# ====== Load model ======
chat_model = genai.GenerativeModel('gemini-2.5-flash')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_loaded = False
executor = ThreadPoolExecutor(max_workers=4)

# Configuration
DRIVE_URL = "https://drive.google.com/uc?id=1GN47lUF7RUZsjpNh7e0y2MHYlS0lq6SF"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "outputs"
MAX_IMAGE_SIZE = (800, 600)
CLEANUP_INTERVAL = 3600
FRAME_SKIP = 2  # Process every 2nd frame for video
UPLOAD_DIR = "uploads"


# ====== Schema ======
class ChatMessage(BaseModel):
    question: str

# ====== Schema cho Khuyến Nghị ======
class RecommendationRequest(BaseModel):
    fresh_count: int
    rotten_count: int
    
# ====== Đọc file kiến thức ======
def load_mango_knowledge():
    path = "data/mango_knowledge.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Không có dữ liệu mô tả về xoài."

mango_knowledge = load_mango_knowledge()

async def load_model():
    """Load YOLO model asynchronously"""
    global model, model_loaded
    try:
        logger.info("Starting model loading...")
        
        if not os.path.exists(MODEL_PATH):
            logger.info(" Downloading model from Google Drive...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor,
                gdown.download,
                DRIVE_URL,
                MODEL_PATH,
                False
            )
            logger.info(" Model downloaded successfully")

        logger.info(" Loading YOLO model...")
        loop = asyncio.get_event_loop()
        model = await loop.run_in_executor(executor, YOLO, MODEL_PATH)

        logger.info(" Warming up model...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        await loop.run_in_executor(executor, model.predict, dummy_img, 0.5)

        model_loaded = True
        logger.info(" Model loaded and warmed up successfully!")
    except Exception as e:
        logger.error(f" Error loading model: {str(e)}")
        model_loaded = False

def cleanup_old_files():
    """Clean up old generated files"""
    try:
        if os.path.exists(OUTPUT_DIR):
            now = time.time()
            for filename in os.listdir(OUTPUT_DIR):
                file_path = os.path.join(OUTPUT_DIR, filename)
                if os.path.isfile(file_path):
                    if now - os.path.getctime(file_path) > 3600:
                        os.remove(file_path)
                        logger.info(f"Cleaned up old file: {filename}")
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
    logger.info(" Starting FastAPI application...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.create_task(load_model())
    asyncio.create_task(periodic_cleanup())
    yield
    # Shutdown
    logger.info(" Shutting down application...")
    executor.shutdown(wait=True)

app = FastAPI(
    title="Mango Quality Checker API with Video Support",
    description="ML-powered mango detection with image, video, and real-time support",
    version="3.0.0",
    lifespan=lifespan
)

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
    scale = min(max_width / width, max_height / height)
    
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

##chạy model lấy bounding box
def process_frame_sync(img: np.ndarray, conf_threshold: float = 0.5) -> Dict:
    """Process a single image and return detection results + annotated image"""
    try:
        img_resized = resize_image(img)
        results = model.predict(img_resized, conf=conf_threshold, verbose=False)
        boxes = results[0].boxes

        annotated_img = img_resized.copy()
        detections = []

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            raw_label = model.names[cls_id]

            # Chuẩn hóa nhãn
            if "fresh" in raw_label.lower():
                label = "fresh"
                color = (0, 255, 0)
                message = "Xoài ngon rồi đấy"
            else:
                label = "rotten"
                color = (0, 0, 255)
                message = "Ui, xoài hỏng rồi"

            # Lấy tọa độ bbox
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Vẽ khung
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                annotated_img,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            #  IMPORTANT: Ajouter bbox dans les détections
            detections.append({
                "label": label,
                "confidence": round(conf * 100, 2),
                "message": message,
                "bbox": [x1, y1, x2, y2]  # ← AJOUT DES COORDONNÉES
            })

        return {
            "detections": detections,
            "annotated_img": annotated_img
        }

    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise e
    
##Vẽ bounding box lên ảnh
def draw_detections(img: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw bounding boxes on image"""
    annotated_img = img.copy()
    
    for det in detections:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["bbox"]
        
        color = (0, 255, 0) if label == "fresh" else (0, 0, 255)
        
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            annotated_img,
            f"{label} {conf:.1f}% ",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )
    
    return annotated_img

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Mango Quality Checker API with Video Support",
        "model_loaded": model_loaded,
        "version": "3.0.0",
        "features": ["image", "video", "realtime"]
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }

###Dự đoán chất lượng xoài từ 1 ảnh
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Predict mango quality from uploaded image"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait a moment and try again.")

    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        logger.info(f"Processing image: {file.filename}")
        contents = await file.read()

        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_frame_sync, img, 0.5)

        annotated_img = result["annotated_img"]

        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        await loop.run_in_executor(executor, cv2.imwrite, output_path, annotated_img)

        logger.info(f" Processed {len(result['detections'])} detections successfully!")

        return {
            "results": result["detections"],
            "image_url": f"/download/{output_filename}",
            "processing_time": "optimized"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

##Dùng cho webcam / realtime từ frontend
@app.post("/predict-frame/")
async def predict_frame(file: UploadFile = File(...)):
    """Realtime prediction for a single frame (used for webcam stream)."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please wait a moment."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image frame"
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty frame received")

        # Giải mã frame
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Cannot decode frame")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_frame_sync, img, 0.5)

        # Encode ảnh annotated thành base64 (cho frontend hiển thị realtime)
        _, buffer = cv2.imencode(".jpg", result["annotated_img"])
        frame_base64 = base64.b64encode(buffer).decode("utf-8")

        #  Lưu lại frame đã xử lý
        output_filename = f"frame_{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        await loop.run_in_executor(executor, cv2.imwrite, output_path, result["annotated_img"])

        # Trả về kết quả
        logger.info(f" Realtime frame processed with {len(result['detections'])} detections")

        return {
            "detections": result["detections"],
            "frame_base64": frame_base64,  # ảnh realtime
            "image_url": f"/download/{output_filename}",  # ảnh được lưu lại
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Realtime frame error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@app.post("/predict-video/")
async def predict_video(file: UploadFile = File(...)):
    """Analyze mango quality in uploaded video."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model is still loading. Please wait a moment.")

    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    try:
        logger.info(f"🎥 Processing video: {file.filename}")
        
        # Créer le dossier uploads s'il n'existe pas
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty video uploaded")

        # Lưa video tạm
        temp_video_path = os.path.join(UPLOAD_DIR, f"temp_{uuid.uuid4().hex}.mp4")
        with open(temp_video_path, "wb") as f:
            f.write(contents)

        # Đọc video
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            os.remove(temp_video_path)
            raise HTTPException(status_code=400, detail="Cannot open video file")

        # Récupérer les propriétés vidéo
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"📹 Video info: {width}x{height} @ {fps}fps, {total_frames} frames")

        # Préparer le fichier de sortie
        output_filename = f"video_{uuid.uuid4().hex}.mp4"
        output_video_path = os.path.join(OUTPUT_DIR, output_filename)

        # Utiliser h264 codec (plus compatible)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # ou 'avc1' pour h264
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        if not out.isOpened():
            cap.release()
            os.remove(temp_video_path)
            raise HTTPException(status_code=500, detail="Cannot create output video")

        frame_count = 0
        processed_count = 0
        fresh_count = 0
        rotten_count = 0
        detections_by_frame = []

        loop = asyncio.get_event_loop()

        logger.info(" Processing video frames...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            
            # Traiter chaque frame (ou tous les N frames pour optimiser)
            if frame_count % FRAME_SKIP == 0 or frame_count == 1:
                try:
                    result = await loop.run_in_executor(executor, process_frame_sync, frame, 0.5)
                    annotated_frame = result["annotated_img"]
                    
                    # S'assurer que la frame a les bonnes dimensions
                    if annotated_frame.shape[:2] != (height, width):
                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                    
                    out.write(annotated_frame)
                    processed_count += 1

                    # Compter les détections
                    if result["detections"]:
                        frame_fresh = sum(1 for d in result["detections"] if d["label"] == "fresh")
                        frame_rotten = sum(1 for d in result["detections"] if d["label"] == "rotten")
                        
                        fresh_count += frame_fresh
                        rotten_count += frame_rotten

                        # Garder seulement les 50 premières frames avec détections
                        if len(detections_by_frame) < 50:
                            detections_by_frame.append({
                                "frame": frame_count,
                                "time": round(frame_count / fps, 2),
                                "detections": result["detections"]
                            })
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    # Écrire la frame originale en cas d'erreur
                    out.write(frame)
            else:
                # Frames non traitées : écrire l'original
                out.write(frame)

            # Log progression
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count}/{total_frames} frames...")

        # Libérer les ressources
        cap.release()
        out.release()
        
        # Supprimer le fichier temporaire
        try:
            os.remove(temp_video_path)
        except:
            pass

        # Vérifier que le fichier de sortie existe et n'est pas vide
        if not os.path.exists(output_video_path) or os.path.getsize(output_video_path) == 0:
            raise HTTPException(status_code=500, detail="Failed to create output video")

        logger.info(f" Video processed: {processed_count} frames analyzed, {fresh_count} fresh, {rotten_count} rotten")

        # Structure de réponse compatible avec le frontend
        return {
            "message": "Video processed successfully",
            "summary": {
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "total_detections": fresh_count + rotten_count,
                "fresh_count": fresh_count,
                "rotten_count": rotten_count
            },
            "detections_by_frame": detections_by_frame,
            "video_url": f"/download/{output_filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        # Nettoyer les fichiers temporaires
        try:
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        except:
            pass
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed image"""
    filename = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, filename)
    
    if os.path.exists(file_path):
        return FileResponse(
            file_path,
            media_type="image/jpeg",
            filename=filename,
            headers={"Cache-Control": "max-age=3600"}
        )
    
    raise HTTPException(status_code=404, detail="File not found")

@app.delete("/cleanup")
async def manual_cleanup():
    """Manual cleanup endpoint"""
    try:
        cleanup_old_files()
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False
    )

    # ====== API Chatbot ======
@app.post("/chat/")
async def chat_with_bot(message: ChatMessage):
    """
    Chatbot về trái cây, trong đó xoài là chủ đề chuyên sâu nhất.
    """
    user_question = message.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống.")

    prompt = f"""
    Bạn là một chuyên gia về trái cây nhiệt đới, đặc biệt am hiểu về xoài.
    Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng liên quan đến:
    - Xoài (các giống xoài, cách chọn xoài ngon, giá trị dinh dưỡng, cách bảo quản, lợi ích, nguồn gốc...)
    - Các loại trái cây khác (chuối, cam, táo, dưa hấu, v.v.), với độ chi tiết vừa phải.

    ---KIẾN THỨC THAM KHẢO VỀ XOÀI---
    {mango_knowledge}
    ----------------------------------

    Hướng dẫn trả lời:
    1. Nếu câu hỏi liên quan đến xoài → trả lời chi tiết, có ví dụ cụ thể, chính xác về giống, vùng trồng, dinh dưỡng, hoặc cách phân biệt.
    2. Nếu câu hỏi về trái cây khác → trả lời ngắn gọn, dễ hiểu, nhưng vẫn đảm bảo đúng kiến thức.
    3. Nếu người dùng hỏi chung (ví dụ: “trái cây nào tốt cho da?”) → so sánh nhẹ và ưu tiên nhắc đến xoài nếu phù hợp.
    4. Nếu bạn không chắc chắn, hãy nói: “Mình chưa có đủ thông tin chính xác để trả lời phần này.”

    Trả lời ngắn gọn, thân thiện, bằng tiếng Việt, như một người hướng dẫn nông sản.
    -------------------
    Câu hỏi: {user_question}
    """

    try:
        response = chat_model.generate_content(prompt)
        return {"answer": response.text}
    except Exception as e:
        print(f"Chatbot error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi truy vấn chatbot.")



@app.post("/get-recommendation/")
async def get_recommendation(request: RecommendationRequest):
    """
    Tự động tạo khuyến nghị dựa trên kết quả phát hiện xoài
    """
    fresh = request.fresh_count
    rotten = request.rotten_count
    total = fresh + rotten

    if total == 0:
        return {"recommendation": "Không phát hiện được xoài nào để đưa ra khuyến nghị."}

    prompt = f"""
    Bạn là chuyên gia về xoài và chế biến thực phẩm.
    
    Kết quả phân tích:
    - Tổng số xoài: {total} quả
    - Xoài tươi ngon: {fresh} quả ({round(fresh/total*100, 1)}%)
    - Xoài đã hỏng: {rotten} quả ({round(rotten/total*100, 1)}%)

    ---KIẾN THỨC THAM KHẢO VỀ XOÀI---
    {mango_knowledge}
    ----------------------------------

    Hãy đưa ra khuyến nghị CHI TIẾT và THỰC TÊ theo các trường hợp:

    **Nếu chỉ có xoài tươi (rotten = 0):**
    - Gợi ý 2-3 món ăn/đồ uống ngon từ xoài tươi (sinh tố, xoài lắc, salad, kem xoài...)
    - Cách bảo quản để giữ tươi lâu
    - Lợi ích dinh dưỡng

    **Nếu có cả tươi và hỏng:**
    - Xoài tươi: gợi ý món ăn
    - Xoài hỏng: CẢNH BÁO không nên ăn, giải thích tại sao, và hướng dẫn cách xử lý (bỏ đi an toàn, không làm phân vì có thể chứa nấm bệnh)

    **Nếu toàn bộ đều hỏng (fresh = 0):**
    - Cảnh báo nghiêm túc về nguy cơ sức khỏe
    - Khuyên KHÔNG sử dụng
    - Hướng dẫn cách nhận biết xoài tươi lần sau

    Trả lời bằng tiếng Việt, thân thiện nhưng CHÍNH XÁC về mặt an toàn thực phẩm.
    Độ dài: 4-6 câu, súc tích, dễ hiểu.
    """

    try:
        response = chat_model.generate_content(prompt)
        return {"recommendation": response.text}
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        # Fallback recommendation
        if rotten == 0:
            fallback = f"🎉 Bạn có {fresh} quả xoài tươi ngon! Có thể làm sinh tố xoài, xoài lắc muối ớt, hoặc ăn trực tiếp. Bảo quản trong ngăn mát tủ lạnh để giữ tươi lâu hơn."
        elif fresh == 0:
            fallback = f"⚠️ Cả {rotten} quả xoài đều đã hỏng. Không nên sử dụng vì có thể gây hại cho sức khỏe. Hãy chọn xoài có vỏ căng mịn, không có vết thâm đen lần sau nhé!"
        else:
            fallback = f"Bạn có {fresh} xoài tươi và {rotten} xoài hỏng. Sử dụng xoài tươi để chế biến món ăn, còn xoài hỏng nên loại bỏ để đảm bảo an toàn."
        
        return {"recommendation": fallback}