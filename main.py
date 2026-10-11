import os
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
# ==========================================
# 1. LOGGING
# ==========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecobite")
# ==========================================
# 2. FASTAPI APPLICATION
# ==========================================
app = FastAPI(
    title="EcoBite AI",
    description="AI Leftover Food Analysis API",
    version="1.0.0"
)
# ==========================================
# 3. CORS CONFIGURATION
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ==========================================
# 4. GEMINI CONFIGURATION
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = None
if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )
    logger.info("Gemini API key detected.")
else:
    logger.warning("GEMINI_API_KEY is missing.")
# ==========================================
# 5. HOME ENDPOINT
# ==========================================
@app.get("/")
async def home():
    return {
        "success": True,
        "message": "EcoBite AI backend is running."
    }
# ==========================================
# 6. HEALTH CHECK
# ==========================================
@app.get("/health")
async def health():
    return {
        "success": True,
        "gemini_configured": client is not None
    }
# ==========================================
# 7. ANALYZE LEFTOVER FOOD
# ==========================================
@app.post("/analyze")
async def analyze_food(
    image: UploadFile = File(...)
):
    # Check Gemini API configuration
    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured."
        )
    # Validate uploaded image
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image."
        )
    try:
        # Read uploaded image
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image is empty."
            )
        logger.info(
            "Received image: %s, type: %s, size: %s bytes",
            image.filename,
            image.content_type,
            len(image_bytes)
        )
        # Prompt for Gemini
        prompt = """
You are EcoBite AI, an assistant that helps reduce food waste.
Analyze the uploaded image and provide the following:
1. FOOD IDENTIFICATION
Identify the visible food. If uncertain, explain the uncertainty.
2. ESTIMATED LEFTOVER MASS
Estimate the mass of the visible leftover food in grams.
Give a reasonable approximate range if an exact estimate is not possible.
3. ESTIMATE LIMITATION
Clearly explain that the weight is a visual estimate only.
The actual weight must be verified using a weighing scale.
Do not claim that image analysis provides an exact measurement.
4. FOOD WASTE REDUCTION TIP
Provide one short, practical suggestion for reducing food waste.
If the image does not show food clearly, say so instead of guessing.
Use clear, simple English with headings.
"""
        # Prepare Gemini contents
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=prompt
                    ),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=image.content_type
                    )
                ]
            )
        ]
        logger.info("Sending image to Gemini.")
        # Generate analysis
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents
        )
        # Extract response text
        analysis_text = response.text
        # Log response metadata without logging image contents
        logger.info(
            "Gemini response received. Text available: %s",
            bool(analysis_text and analysis_text.strip())
        )
        # Handle empty response
        if not analysis_text or not analysis_text.strip():
            logger.error(
                "Gemini returned no text. Finish reason: %s",
                getattr(
                    getattr(response, "candidates", [None])[0],
                    "finish_reason",
                    None
                ) if getattr(response, "candidates", None) else None
            )
            raise HTTPException(
                status_code=502,
                detail=(
                    "Gemini returned no text. "
                    "Check the backend logs for response details."
                )
            )
        # Return successful analysis
        return {
            "success": True,
            "filename": image.filename,
            "analysis": analysis_text.strip()
        }
    except HTTPException:
        raise
    except Exception as error:
        logger.exception(
            "Gemini analysis failed: %s",
            type(error).__name__
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"AI analysis failed: "
                f"{type(error).__name__}: {str(error)}"
            )
        )
    finally:
        await image.close()
