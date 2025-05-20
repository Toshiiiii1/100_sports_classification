from fastapi import FastAPI, UploadFile, HTTPException, File
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import config
import keras
import time

app = FastAPI()
clf_model = keras.saving.load_model(config.model_path)
MAX_SIZE = 5 * 1024 * 1024

# define API repsone schema
class PredictRepsone(BaseModel):
    label: str
    confidence: float
    time: float

@app.get("/")
async def root():
    return {"message": "API works"}

@app.post("/predict")
async def predict(file: UploadFile):
    # validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="only except image file (png/jpg)")

    # check file size
    if file.size > MAX_SIZE:
        raise HTTPException(status_code=413, detail="uploaded file too large (limit: 5MB)")
    
    # read image as bytes type
    img_bytes = io.BytesIO(await file.read())
    # convert image bytes type to np array
    image = Image.open(img_bytes).resize((config.img_width, config.img_heigth))
    img_array = keras.utils.img_to_array(image)
    batch_img = np.array([img_array]) # [1, 224, 224]
    
    # handle prediction fail
    try:
        start_time = time.time()
        predictions = clf_model.predict(batch_img) # [1, 100]
        end_time = time.time()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    label_index = np.argmax(predictions, axis=1)[0]
    label_prob = predictions[0][label_index] # np float32
    label = config.int2label[label_index]
    infer_time = (end_time - start_time) # second
    
    return PredictRepsone(
        label=label,
        confidence=round(float(label_prob), 3),
        time=round(infer_time, 3),
    )