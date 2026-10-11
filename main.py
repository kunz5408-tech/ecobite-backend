import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from google import genai
from google.genai import types

app = FastAPI()

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from environment variables")

client = genai.Client(api_key=api_key)


@app.get("/")
def home():
    return {"message": "EcoBite backend is running!"}


@app.post("/analyze")
async def analyze_food(image: UploadFile = File(...)):

    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image"
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty"
        )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_text(
                    text=(
                        "Analyze this leftover food image. "
                        "Identify the food and estimate its leftover "
                        "mass in grams. State that the weight is only "
                        "a visual estimate and must be verified using a scale."
                    )
                ),
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=image.content_type
                )
            ]
        )

        return {
            "success": True,
            "analysis": response.text
        }

    except Exception as e:
        print(f"Gemini API error: {type(e).__name__}: {e}")

        raise HTTPException(
            status_code=502,
            detail=f"AI analysis failed: {type(e).__name__}: {str(e)}"
        )
