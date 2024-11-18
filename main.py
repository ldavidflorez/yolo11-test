from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse
import cv2
import math
import ultralytics
import uvicorn
import asyncio
from uuid import uuid4
import os
from fastapi.staticfiles import StaticFiles

# Cargar el modelo YOLO
model = ultralytics.YOLO("yolo11n.pt")

# Inicializar FastAPI
app = FastAPI()

# Servir archivos estáticos desde 'uploads'
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Crear carpeta para subir videos
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


async def video_frame_generator(file_path: str, websocket: WebSocket):
    """Generador de frames desde un video."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        await websocket.send_text("Error: Cannot open video file.")
        return

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            for r in results:
                for box in r.boxes:
                    # Coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Calcular la confianza y el texto de la clase
                    confidence = math.ceil((box.conf[0] * 100))
                    label = r.names[int(box.cls[0])]

                    # Imprimir resultados (opcional, para depuración)
                    print(
                        f"LABEL: {label}  COORDINATES: {(x1, y1)}, { (x2, y2)}  CONFIDENCE: {confidence}%"
                    )

            # Codificar el frame a JPEG
            _, buffer = cv2.imencode(".jpg", results[0].plot())
            frame_bytes = buffer.tobytes()

            # Enviar el frame como datos binarios a través del WebSocket
            await websocket.send_bytes(frame_bytes)

            # Añadir un pequeño retraso para evitar saturar al cliente
            await asyncio.sleep(0.01)
    finally:
        cap.release()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Subir un archivo de video."""
    unique_filename = f"{uuid4().hex}.mp4"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"message": "Video uploaded successfully", "filename": unique_filename}


@app.websocket("/ws/{filename}")
async def websocket_endpoint(websocket: WebSocket, filename: str):
    """WebSocket para transmitir frames del video procesado."""
    await websocket.accept()

    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        await websocket.send_text("Error: File not found.")
    else:
        await video_frame_generator(file_path, websocket)

    await websocket.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    """Página de inicio con una interfaz gráfica."""
    html_content = open("templates/index.html").read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
