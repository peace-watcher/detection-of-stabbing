from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from groundingdino.util.inference import load_model, load_image, predict, annotate
import uvicorn
import cv2
import base64
import numpy as np
from PIL import Image
import io
from pathlib import Path

base_path = Path(__file__).resolve().parent.parent

config_path = base_path / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = base_path / "models/groundingdino_swint_ogc.pth"

config_path_str = str(config_path)
weights_path_str = str(weights_path)

model = load_model(config_path_str, weights_path_str)

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def get():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Real-Time Detection</title>
        <style>
            #videoElement, #outputElement {
                width: 640px;
                height: 480px;
                border: 1px solid black;
            }
        </style>
    </head>
    <body>
        <h1>Real-Time Object Detection</h1>
        <video id="videoElement" autoplay></video>
        <img id="outputElement" />
        <script>
            var video = document.getElementById("videoElement");
            var img = document.getElementById("outputElement");
            var ws = new WebSocket("ws://107.167.183.252:8000/ws");

            ws.onopen = function(event) {
                console.log("WebSocket is open now.");
                startVideo();
            };

            ws.onmessage = function(event) {
                img.src = "data:image/jpeg;base64," + event.data;
            };

            ws.onclose = function(event) {
                console.log("WebSocket is closed now.");
            };

            ws.onerror = function(error) {
                console.log("WebSocket Error: " + error);
            };

            function startVideo() {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        video.srcObject = stream;
                        video.play();
                        sendFrame();
                    })
                    .catch(function(err) {
                        console.log("An error occurred: " + err);
                    });
            }

            function sendFrame() {
                var canvas = document.createElement("canvas");
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                var context = canvas.getContext("2d");

                function captureFrame() {
                    context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    var data = canvas.toDataURL("image/jpeg").split(",")[1];
                    ws.send(data);
                    requestAnimationFrame(captureFrame);
                }

                captureFrame();
            }
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Decode base64 image
            try:
                image_data = base64.b64decode(data)
                image_temp = Image.open(io.BytesIO(image_data))
                image = np.array(image_temp)
            except Exception as e:
                print("Error decoding image:", e)
                continue
            
            # 이미지 처리
            try:
                temp_path = tempfile.mktemp(suffix=".jpg")
                cv2.imwrite(temp_path, image)
                image_source, image = load_image(temp_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image,
                    caption="a person with a knife",
                    box_threshold=0.35,
                    text_threshold=0.25
                )

                # 바운딩 박스 그리기
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
                
                # 이미지 인코딩 및 전송
                _, buffer = cv2.imencode('.jpg', annotated_frame)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                await websocket.send_text(encoded_image)
            except Exception as e:
                print("Error processing image:", e)
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

