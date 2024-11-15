from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import math
import ultralytics
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os

from dotenv import load_dotenv

load_dotenv()


model = ultralytics.YOLO("yolo11l.pt")

# Configuración del texto y los colores
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.7
box_color = (0, 0, 255)  # Color del bounding box
text_color = (255, 255, 255)  # Color del texto
bg_color = (0, 0, 0)  # Color del fondo del texto
box_thickness = 2

app = FastAPI()

# Ruta temporal para almacenar el archivo de video subido
temp_video_path = "uploaded_video.mp4"


def video_frame_generator():
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
                label = f"{r.names[int(box.cls[0])]} {confidence}%"

                # Imprimir resultados
                print(
                    f"LABEL: {r.names[int(box.cls[0])]}  COORDINATES: {(x1, y1)}, { (x2, y2)}  CONFIDENCE: {confidence}%"
                )

                # Dibujar el bounding box con un grosor más grueso
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

                # Medidas del texto
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, fontScale, box_thickness
                )

                # Fondo del texto
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width + 4, y1),
                    bg_color,
                    -1,
                )

                # Colocar el texto encima del fondo
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 5),
                    font,
                    fontScale,
                    text_color,
                    thickness=2,
                )

        # Codificar el frame a JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Formato para streaming de imágenes
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"message": "Video uploaded successfully"}


@app.get("/stream")
async def stream_video():
    return StreamingResponse(
        video_frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
async def index():
    html_content = open("templates/index.html").read()
    return HTMLResponse(content=html_content)


# Get your authtoken from https://dashboard.ngrok.com/get-started/your-authtoken
auth_token = os.getenv("NGROK_TOKEN")

# Set the authtoken
ngrok.set_auth_token(auth_token)

# Connect to ngrok
ngrok_tunnel = ngrok.connect(9000)

# Print the public URL
print("Public URL:", ngrok_tunnel.public_url)

# Apply nest_asyncio
nest_asyncio.apply()

# Run the uvicorn server
uvicorn.run(app, port=9000)
