from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import time
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import os

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Ruta temporal para almacenar el archivo de video subido
temp_video_path = "uploaded_video.mp4"


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    with open(temp_video_path, "wb") as buffer:
        buffer.write(await file.read())
    return {"message": "Video uploaded successfully"}


def video_frame_generator():
    # Utilizar el video subido para el procesamiento
    cap = cv2.VideoCapture(temp_video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Simulación de procesamiento (reemplazar con procesamiento real)
        time.sleep(0.1)
        print("Procesando frame...")

        # Codificar el frame a JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        # Formato para streaming de imágenes
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

    cap.release()


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
