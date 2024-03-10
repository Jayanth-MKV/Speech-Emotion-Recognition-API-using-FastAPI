from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
from ml import process_audio_file, predict

app = FastAPI()


@app.post("/predict_audio")
async def process_audio_emotion(file: UploadFile,):
    content_type = file.content_type

    # Check if the uploaded file is an audio file
    if "audio" not in str(content_type):
        raise HTTPException(
            status_code=415, detail="Unsupported Media Type. Please upload an audio file.")

    audio_bytes = await file.read()
    audio_path = io.BytesIO(audio_bytes)
    predictions_data = await process_audio_file(audio_path)
    # predictions_data = predictions_data.reshape(1, -1, 1)
    print("predictions_data : ", predictions_data)
    print("predictions_data : ", predictions_data[0].shape)
    predictions_list = await predict(predictions_data[0])
    print("predictions_list : ", predictions_list.shape)
    # ind = np.argmax(predictions_list[0])

    return {"emotion": predictions_list.tolist()}
