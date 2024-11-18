from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse
import cv2
import math
import ultralytics
import uvicorn
import asyncio

model = ultralytics.YOLO("yolo11n.pt")

app = FastAPI()

# Ruta temporal para almacenar el archivo de video subido
temp_video_path = "uploaded_video.mp4"


async def video_frame_generator(websocket: WebSocket):
    # Utilizar el video subido para el procesamiento
    cap = cv2.VideoCapture(temp_video_path)
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

                # Imprimir resultados
                print(
                    f"LABEL: {label}  COORDINATES: {(x1, y1)}, { (x2, y2)}  CONFIDENCE: {confidence}%"
                )

        # Codificar el frame a JPEG
        _, buffer = cv2.imencode(".jpg", results[0].plot())
        frame_bytes = buffer.tobytes()

        # Enviar el frame como datos binarios a través del WebSocket
        await websocket.send_bytes(frame_bytes)

        # Añadir un pequeño retraso para evitar saturar al cliente
        await asyncio.sleep(0.03)

    cap.release()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"message": "Video uploaded successfully"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await video_frame_generator(websocket)
    await websocket.close()


@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = open("templates/index.html").read()
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
