from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import librosa
from ml import extract_features

app = FastAPI()


class AudioEmotionRequest(BaseModel):
    file: UploadFile


@app.post("/process_audio")
async def process_audio_emotion(request: AudioEmotionRequest):
    content_type = request.file.content_type

    # Check if the uploaded file is an audio file
    if "audio" not in str(content_type):
        raise HTTPException(
            status_code=415, detail="Unsupported Media Type. Please upload an audio file.")

    # Save the uploaded file to a temporary location
    with open("temp_audio.wav", "wb") as audio_file:
        audio_file.write(request.file.file.read())

    # Extract features from the audio file
    try:
        data, sample_rate = librosa.load(
            "temp_audio.wav", duration=3, offset=0.5, res_type='kaiser_fast')
        features = extract_features(data)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing audio: {str(e)}")

    # You can now use the 'features' array to make predictions using your model
    # For example, if using a pre-trained model:
    # model = load_model("model.h5")
    # predictions = model.predict(features)

    # Replace the next line with your actual emotion prediction logic
    # For now, returning the extracted features as JSON response
    return JSONResponse(content={"features": features.tolist()}, status_code=200)
