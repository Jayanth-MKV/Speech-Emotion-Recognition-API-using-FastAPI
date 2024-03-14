# ml.py
import numpy as np
import librosa
import io
import tensorflow as tf


model = tf.keras.models.load_model('./model/female_model.h5')

m = np.load('./model/mean.npy')
s = np.load('./model/std.npy')

async def extract_features(data):

    result = np.array([])

    # mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=42) #42 mfcc so we get frames of ~60 ms
    mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=58)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    result = np.array(mfccs_processed)

    return result

async def Standardize(x):
     return (x - m) / (s)

async def process_audio_file(file_path):
    data, sample_rate = librosa.load(
        file_path, duration=3, offset=0.5, res_type='kaiser_fast')
    res1 = await extract_features(data)
    
    # Min-Max Scaling
    # min_value = np.min(res1)
    # max_value = np.max(res1)
    # scaled_res1 = (res1 - min_value) / (max_value - min_value)

    scaled_res1 = await Standardize(res1)

    # print("sr - ",scaled_res1)
    result = scaled_res1.reshape(1, 58, 1)
    # print("result - ",result)
    return result

async def predict_audio_file(file):
    audio_bytes = await file.read()
    audio_path = io.BytesIO(audio_bytes)
    predictions_data = await process_audio_file(audio_path)
    predictions_data = predictions_data.reshape(1, -1, 1)
    predictions_list = await model.predict(predictions_data)
    return predictions_list


async def predict(data):
    p = model.predict(data)
    print("p - ",p)
    ind = np.argmax(p[0])
    print("Ind - ",ind)
    return mapper[ind]

mapper = ['fear', 'happy', 'neutral', 'sad']