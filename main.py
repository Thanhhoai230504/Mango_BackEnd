from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import os

app = FastAPI(title="Mango Quality Checker API")

# ✅ Bật CORS cho frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc ["http://localhost:3000", "http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO model
model = YOLO("best.pt")

# Thư mục lưu ảnh kết quả
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Đọc ảnh từ upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Chạy dự đoán với YOLO
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
