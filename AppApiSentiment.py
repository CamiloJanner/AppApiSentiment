from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from deep_translator import GoogleTranslator
import numpy as np
import pickle
import os
import random
import uvicorn

app = FastAPI()

# Paths locales en la instancia EC2
MODEL_PATH = "modelo_sentimiento.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# Cargar modelo y tokenizador al iniciar
try:
    model = load_model(MODEL_PATH, compile=False)
    with open(TOKENIZER_PATH, "rb") as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    print(f"Error cargando modelo o tokenizer: {str(e)}")
    raise e

# Respuestas personalizadas por clase
responses = {
    0: [
        "¡Parece que estás de buen ánimo! Sigue disfrutando tu día. 😊",
        "Tu mensaje refleja una actitud positiva. ¡Sigue así! 🌟",
        "Se nota optimismo en tus palabras. ¡Eso es genial! 💪"
    ],
    1: [
        "Tu mensaje parece ser neutral, sin una emoción fuerte asociada. 🤔",
        "No detecto un sentimiento marcado en tu mensaje. ¿Tienes algo en mente? 🧐",
        "Parece que es un comentario equilibrado, sin inclinación emocional. 🎭"
    ],
    2: [
        "Percibo que podrías estar sintiéndote mal. Si necesitas hablar, aquí estoy. 🖤",
        "Tu mensaje suena algo negativo. Espero que todo mejore pronto. 🌧️",
        "Parece que no estás en tu mejor día. Recuerda que todo pasa. 💙"
    ]
}

# Endpoint principal
@app.post("/predict/")
async def predict_sentiment(request: Request):
    try:
        data = await request.json()
        user_text = data.get("text", "").strip()

        if not user_text:
            return JSONResponse(content={"error": "Texto vacío o no enviado."}, status_code=400)

        # Traducir de español a inglés
        translated_text = GoogleTranslator(source='es', target='en').translate(user_text)

        # Procesar entrada
        sequence = tokenizer.texts_to_sequences([translated_text])
        padded = pad_sequences(sequence, maxlen=100, padding="post", truncating="post")
        prediction = model.predict(padded)
        score = prediction[0][0]

        if score < 0.4:
            sentiment_class = 2  # Negativo
        elif score > 0.6:
            sentiment_class = 0  # Positivo
        else:
            sentiment_class = 1  # Neutro

        response_text = random.choice(responses.get(sentiment_class, ["Error: Clase fuera de rango."]))
        sentimiento_nombre = ["Positivo", "Neutro", "Negativo"][sentiment_class]

        return JSONResponse(content={
            "sentimiento": sentimiento_nombre,
            "respuesta": response_text,
            "score": float(score)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Para ejecutar directamente si se corre como script
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
