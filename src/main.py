from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse
from groundingdino.util.inference import load_model, load_image, predict, annotate
import uvicorn
import cv2
import base64
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path



base_path = Path(__file__).resolve().parent.parent

config_path = base_path / "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
weights_path = base_path / "models/groundingdino_swint_ogc.pth"

config_path_str = str(config_path)
weights_path_str = str(weights_path)


model = load_model(config_path_str, weights_path_str)

app = FastAPI()

# HTML 프론트엔드
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Real-Time Detection</title>
</head>
<body>
<h1>Real-Time Object Detection</h1>
<img id="videoElement">
<script>
    var img = document.getElementById("videoElement");
    var ws = new WebSocket("ws://107.167.183.252:8000/ws");
    ws.onmessage = function(event) {
        img.src = "data:image/jpeg;base64," + event.data;
    };
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get():
    return html

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture(0)  # 웹캠 열기

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 프레임을 임시 파일로 저장
            temp_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(temp_path, frame)
            
            # 프레임 이미지 처리
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
    
    finally:
        cap.release()
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

