import os
import sqlite3
from datetime import datetime
import io
from sqlalchemy import create_engine

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import tensorflow as tf
from PIL import Image
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent

# --- Environment Variable Setup ---
# Set your Google AI API key. Get it from https://aistudio.google.com/app/apikey
# It's best to set this in your terminal before running the script:
# On Windows: set GOOGLE_API_KEY=your_key_here
# On macOS/Linux: export GOOGLE_API_KEY=your_key_here
if 'GOOGLE_API_KEY' not in os.environ:
    print("WARNING: Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

# --- Constants and Model Loading ---
MODEL_PATH = 'defect_detector.h5'
DB_PATH = 'qc_database.db'
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 150
CLASS_NAMES = ['def_front', 'ok_front'] # Based on the folder names from training

# Load the trained computer vision model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Defect detection model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized.")

# --- FastAPI App Initialization ---
app = FastAPI(title="Manufacturing Quality Control API")
init_db() # Initialize the database when the app starts

# --- Pydantic Models for API ---
class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float

class Query(BaseModel):
    text: str

# --- Prediction Endpoint ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    image = image.resize((IMAGE_HEIGHT, IMAGE_WIDTH))
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make prediction
    predictions = model.predict(img_array)
    score = float(predictions[0][0])
    
    # Determine class and confidence
    if score < 0.5:
        prediction_class = 'def_front'
        confidence = 1 - score
    else:
        prediction_class = 'ok_front'
        confidence = score

    # Save result to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO predictions (filename, prediction, confidence, timestamp) VALUES (?, ?, ?, ?)",
        (file.filename, prediction_class, confidence, datetime.now())
    )
    conn.commit()
    conn.close()

    return {
        "filename": file.filename,
        "prediction": prediction_class,
        "confidence": confidence
    }

# --- Q&A Endpoint (Corrected) ---
@app.post("/ask", response_model=dict)
async def ask_database(query: Query):
    try:
        # Correctly create a SQLAlchemy engine
        engine = create_engine(f"sqlite:///{DB_PATH}")
        db = SQLDatabase(engine=engine)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

        agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

        result = agent_executor.invoke(query.text)
        return {"answer": result['output']}
    except Exception as e:
        # Be sure to import HTTPException from fastapi at the top of your file
        raise HTTPException(status_code=500, detail=str(e))