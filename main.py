import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Vigi-Link API is Live"}

@app.post("/split")
async def split_image(file: UploadFile = File(...)):
    # Read the uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    
    h, w = binary.shape
    share1 = np.random.randint(0, 2, (h, w), dtype=np.uint8) * 255
    share2 = np.zeros_like(share1)

    for i in range(h):
        for j in range(w):
            if binary[i, j] == 255: # White pixel
                share2[i, j] = share1[i, j]
            else: # Black pixel
                share2[i, j] = 255 - share1[i, j]
                
    # In a real app, you'd return both files. For this demo, we return one share.
    _, encoded_img = cv2.imencode('.png', share1)
    return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/png")
