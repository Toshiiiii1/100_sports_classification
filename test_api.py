import pytest
from fastapi.testclient import TestClient # conmunicate direcly to FastAPi app
from api import app
import os
from PIL import Image
import io
import numpy as np

client = TestClient(app)

def create_test_image(size=(224, 224)):
    """Generate a dummy test image"""
    img_array = np.random.rand(*size, 3) * 255
    img = Image.fromarray(img_array.astype('uint8')).convert('RGB')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def test_image():
    return create_test_image()

@pytest.fixture
def valid_image():
    with open("data/test_images/Soccer-at-Coliseum.jpg", "rb") as f:
        return f.read()

def test_healthcheck():
    response = client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"message": "API works"}

def test_predict_valid_image(test_image):
    files = {"file": ("test.jpg", test_image, "image/jpg")}
    response = client.post("/predict", files=files)
    print(response)
    assert response.status_code == 200
    assert "label" in response.json()
    assert "confidence" in response.json()
    assert "time" in response.json()

def test_predict_invalid_file_type():
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "only except image file" in response.json()["detail"]

def test_predict_large_file(test_image):
    size = (3000, 3000)
    img_array = (np.random.rand(*size, 3) * 255).astype('uint8')
    img = Image.fromarray(img_array)
    
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    files = {"file": ("large.png", img_bytes, "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 413
    assert "uploaded file too large" in response.json()["detail"]