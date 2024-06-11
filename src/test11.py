from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from groundingdino.util.inference import load_model, load_image, predict, annotate
import uvicorn
import cv2
import base64
import numpy as np
from PIL import Image
import io
import tempfile
from pathlib import Path
from collections import deque

base_path = Path(__file__).resolve().parent.parent

config_path = base_path / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = base_path / "models/groundingdino_swint_ogc.pth"

config_path_str = str(config_path)
weights_path_str = str(weights_path)

model = load_model(config_path_str, weights_path_str)

app = FastAPI()

# 탐지 결과를 저장하기 위한 큐 (최대 길이 10)
detections_queue = deque(maxlen=10)

# ALERT가 이미 전송되었는지 여부를 저장하는 변수
alert_sent = False

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
    global alert_sent  # 전역 변수로 선언

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            # 디코딩할 데이터의 길이 출력
            print(f"Received data length: {len(data)}")
            
            # 수신된 데이터 일부 출력
            print(f"Received data (first 100 chars): {data[:100]}")

            try:
                # Base64 디코딩
                image_data = base64.b64decode(data)
                print(f"Decoded data length: {len(image_data)}")

                # 디코딩된 데이터 확인
                with open("received_image.jpg", "wb") as f:
                    f.write(image_data)

                # 이미지로 변환
                image = Image.open(io.BytesIO(image_data))
                image = image.convert("RGB")  # 이미지 색상 공간을 RGB로 변환

                # PIL 이미지를 OpenCV 이미지로 변환
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # 색상 공간 변환

                print(f"Image shape: {image.shape}")

            except Exception as e:
                print("Error decoding image:", e)
                continue

            # 이미지 처리
            try:
                temp_path = tempfile.mktemp(suffix=".jpg")
                cv2.imwrite(temp_path, image)
                image_source, image_tensor = load_image(temp_path)
                boxes, logits, phrases = predict(
                    model=model,
                    image=image_tensor,
                    caption="a person with a knife, fallen person, person running away",
                    box_threshold=0.35,
                    text_threshold=0.25
                )

                # 바운딩 박스 그리기
                annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

                # 탐지 결과를 큐에 추가
                detections_queue.append("a knife" in phrases)

                # 최근 8프레임 모두에서 "knife" 탐지 시 ALERT 전송 (한 번만)
                if not alert_sent and detections_queue.count(True) == 10:
                    await websocket.send_text("ALERT")
                    alert_sent = True

                # OpenCV 이미지를 PIL 이미지로 변환
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                annotated_frame = Image.fromarray(annotated_frame)

                # 이미지 인코딩 및 전송
                buffered = io.BytesIO()
                annotated_frame.save(buffered, format="JPEG")
                encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                await websocket.send_text(encoded_image)
            except Exception as e:
                print("Error processing image:", e)
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

