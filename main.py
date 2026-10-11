import os
import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai

app = FastAPI()

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

@app.get("/")
def home():
    return {"message": "EcoBite backend is running!"}

@app.post("/analyze")
async def analyze_food(image: UploadFile = File(...)):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image")

    image_bytes = await image.read()

    try:
        result = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Analyze this leftover food image. Identify the food "
                "and estimate its leftover mass in grams. State that "
                "the weight is only a visual estimate and must be "
                "verified using a scale.",
                {
                    "mime_type": image.content_type,
                    "data": base64.b64encode(image_bytes).decode("utf-8")
                }
            ]
        )
        return {"success": True, "analysis": result.text}
    except Exception as e:
        raise HTTPException(502, f"AI analysis failed: {str(e)}")
