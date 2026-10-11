import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
# ==========================================
# 1. CREATE FASTAPI APP
# ==========================================
app = FastAPI(
    title="EcoBite AI API",
    description="AI-powered leftover food analysis",
    version="1.0.0"
)
# ==========================================
# 2. ENABLE CORS
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# 3. CONNECT TO GEMINI API
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)
# ==========================================
# 4. HOME ENDPOINT
# ==========================================
@app.get("/")
async def home():
    return {
        "success": True,
        "message": "EcoBite AI API is running!"
    }
# ==========================================
# 5. HEALTH CHECK
# ==========================================
@app.get("/health")
async def health_check():
    return {
        "success": True,
        "gemini_configured": client is not None
    }
# ==========================================
# 6. ANALYZE LEFTOVER FOOD
# ==========================================
@app.post("/analyze")
async def analyze_food(image: UploadFile = File(...)):
    # Check API key
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is missing from environment variables."
        )
    # Check uploaded file
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )
    try:
        # Read image
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )
        # Prepare image MIME type
        mime_type = image.content_type
        # Prepare prompt
        prompt = """
        You are EcoBite AI, an assistant that analyzes leftover food.
        Analyze the uploaded image and provide:
        1. Food identification:
           Identify the visible food as accurately as possible.
        2. Estimated leftover mass:
           Estimate the mass of the visible leftover food in grams.
        3. Confidence and limitations:
           Explain that the mass is only a visual estimate.
           The actual weight must be verified using a weighing scale.
           If the image does not provide enough information for a
           reasonable estimate, clearly state this limitation.
        4. Food waste advice:
           Give one short, practical suggestion to reduce food waste.
        Important:
        - Do not claim that the estimated weight is an exact measurement.
        - Do not invent details that cannot be determined from the image.
        - Keep the response clear and easy to understand.
        """
        # Send image and prompt to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=prompt
                        ),
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=mime_type
                        )
                    ]
                )
            ]
        )
        # Get AI response
        analysis_text = response.text
        if not analysis_text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response."
            )
        # Return result
        return {
            "success": True,
            "filename": image.filename,
            "food_type": "See analysis",
            "analysis": analysis_text
        }
    except HTTPException:
        raise
    except Exception as error:
        print(
            f"Gemini API error: {type(error).__name__}: {error}"
        )
        raise HTTPException(
            status_code=502,
            detail=f"AI analysis failed: {type(error).__name__}: {error}"
        )
