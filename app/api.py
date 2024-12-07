import os
import logging
from dotenv import load_dotenv
import google.generativeai as genai
import PIL.Image
import json
import re
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
import requests

# Suppress warning messages
logging.getLogger().setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize FastAPI
app = FastAPI(
    title="Product Information Extraction API",
    description="API for extracting product information from images using Google's Gemini model",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables and configure Gemini
load_dotenv()
key = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)
model = genai.GenerativeModel('gemini-1.5-flash')

def clean_json_data(data):
    if isinstance(data, dict):
        return {clean_json_data(k): clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(x) for x in data]
    elif isinstance(data, str):
        cleaned_text = re.sub(r'\\n|\n', ' ', data)
        cleaned_text = ' '.join(cleaned_text.split())
        return cleaned_text
    else:
        return data

def process_response_to_json(content):
    try:
        content = content.replace('```json', '').replace('```', '').strip()
        content = re.sub(r'\\n|\n', ' ', content)
        json_data = json.loads(content)
        cleaned_json_data = clean_json_data(json_data)
        return cleaned_json_data
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

async def process_image(image: PIL.Image.Image):
    prompt = '''I want the 
    - Product name 
    - Size 
    - Nutrition Facts
    - Vitamins 
    - Ingredients 
    - Not recommended
    - Allergies
    - Any other relevant field

    Only print the field(s) that are available in the image.
    in a proper JSON format.
    '''
    
    try:
        response = model.generate_content([prompt, image])
        json_result = process_response_to_json(response.text)
        return json_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Product Information Extractor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .upload-box {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            border-radius: 4px;
            cursor: pointer;
            margin-bottom: 20px;
        }
        .upload-box:hover {
            border-color: #666;
        }
        #preview {
            max-width: 300px;
            max-height: 300px;
            margin-top: 10px;
            display: none;
        }
        #response {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
            display: none;
            margin-top: 20px;
            font-family: monospace;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Product Information Extractor</h1>
        <div class="upload-box" onclick="document.getElementById('fileInput').click()">
            <p>Click to upload an image or drag and drop here</p>
            <input type="file" id="fileInput" accept="image/*" style="display: none">
            <img id="preview" alt="Preview">
        </div>

        <div class="loading">
            <div class="spinner"></div>
            <p>Processing image...</p>
        </div>

        <div id="response"></div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        const response = document.getElementById('response');
        const loading = document.querySelector('.loading');
        const uploadBox = document.querySelector('.upload-box');

        uploadBox.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#666';
        });

        uploadBox.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#ccc';
        });

        uploadBox.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadBox.style.borderColor = '#ccc';
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFile(file);
            }
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                handleFile(file);
            }
        });

        function handleFile(file) {
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);

            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);

            // Show loading spinner
            loading.style.display = 'block';
            response.style.display = 'none';

            // Send to API
            fetch('/process_image_file', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                response.textContent = JSON.stringify(data, null, 2);
                response.style.display = 'block';
            })
            .catch(error => {
                response.textContent = 'Error: ' + error.message;
                response.style.display = 'block';
            })
            .finally(() => {
                loading.style.display = 'none';
            });
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Your existing endpoints remain the same
@app.post("/process_image_file")
async def process_image_file(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_content = await file.read()
        image = PIL.Image.open(BytesIO(image_content))
        
        result = await process_image(image)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_image_url")
async def process_image_url(image_url: str):
    try:
        response = requests.get(image_url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
        
        image = PIL.Image.open(BytesIO(response.content))
        result = await process_image(image)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_local_image")
async def process_local_image(image_path: str):
    try:
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image file not found")
        
        image = PIL.Image.open(image_path)
        result = await process_image(image)
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))