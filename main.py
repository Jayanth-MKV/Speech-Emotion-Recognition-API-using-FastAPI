from fastapi import FastAPI, Request, UploadFile,File, HTTPException
from fastapi.responses import HTMLResponse
import io
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from ml import process_audio_file, predict,malepredict,femalepredict
import logging
from fastapi import   status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
	exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
	logging.error(f"{request}: {exc_str}")
	content = {'status_code': 10422, 'message': exc_str, 'data': None}
	return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(request=request, name="index.html",)


@app.post("/predict_audio")
async def process_audio_emotion(file: UploadFile= File(...)):
    print(file)
    content_type = file.content_type

    print(content_type)
    # Check if the uploaded file is an audio file
    if "audio" not in str(content_type):
        raise HTTPException(
            status_code=415, detail="Unsupported Media Type. Please upload an audio file.")

    audio_bytes = await file.read()
    # print(audio_bytes)
    with open("received_audio.wav", "wb") as f:
        f.write(audio_bytes)
    audio_path = io.BytesIO(audio_bytes)
    print(audio_path)
    predictions_data = await process_audio_file(audio_path)

    # print("predictions_data : ", predictions_data.shape)

    # Ensure that predictions_data has the correct shape (1, 58, 1)
    if predictions_data.shape != (1, 58, 1):
        raise HTTPException(
            status_code=500, detail=f"Invalid shape of processed audio data: {predictions_data.shape}")

    predictions_list = await predict(predictions_data)
    print("predictions_list : ", predictions_list)
    # ind = np.argmax(predictions_list[0])

    return {"emotion": predictions_list}


@app.post("/predict_audio/male")
async def process_audio_emotion(file: UploadFile= File(...)):
    print(file)
    content_type = file.content_type

    print(content_type)
    # Check if the uploaded file is an audio file
    if "audio" not in str(content_type):
        raise HTTPException(
            status_code=415, detail="Unsupported Media Type. Please upload an audio file.")

    audio_bytes = await file.read()
    # print(audio_bytes)
    with open("received_audio.wav", "wb") as f:
        f.write(audio_bytes)
    audio_path = io.BytesIO(audio_bytes)
    print(audio_path)
    predictions_data = await process_audio_file(audio_path)

    # print("predictions_data : ", predictions_data.shape)

    # Ensure that predictions_data has the correct shape (1, 58, 1)
    if predictions_data.shape != (1, 58, 1):
        raise HTTPException(
            status_code=500, detail=f"Invalid shape of processed audio data: {predictions_data.shape}")

    predictions_list = await malepredict(predictions_data)
    print("predictions_list : ", predictions_list)
    # ind = np.argmax(predictions_list[0])

    return {"emotion": predictions_list}


@app.post("/predict_audio/female")
async def process_audio_emotion(file: UploadFile= File(...)):
    print(file)
    content_type = file.content_type

    print(content_type)
    # Check if the uploaded file is an audio file
    if "audio" not in str(content_type):
        raise HTTPException(
            status_code=415, detail="Unsupported Media Type. Please upload an audio file.")

    audio_bytes = await file.read()
    # print(audio_bytes)
    with open("received_audio.wav", "wb") as f:
        f.write(audio_bytes)
    audio_path = io.BytesIO(audio_bytes)
    print(audio_path)
    predictions_data = await process_audio_file(audio_path)

    # print("predictions_data : ", predictions_data.shape)

    # Ensure that predictions_data has the correct shape (1, 58, 1)
    if predictions_data.shape != (1, 58, 1):
        raise HTTPException(
            status_code=500, detail=f"Invalid shape of processed audio data: {predictions_data.shape}")

    predictions_list = await femalepredict(predictions_data)
    print("predictions_list : ", predictions_list)
    # ind = np.argmax(predictions_list[0])

    return {"emotion": predictions_list}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)