from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import fal_client
import google.generativeai as genai
import os
import shutil

# אתחול האפליקציה
app = FastAPI()

# קריאת מפתחות מהסביבה (נגדיר אותם ב-Render)
os.environ["FAL_KEY"] = os.getenv("FAL_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# מודל לבקשת יצירת תמונה
class GenerationRequest(BaseModel):
    prompt: str
    lora_url: str  # הלינק לקובץ הזהות של המשתמש
    trigger_word: str = "OHAD_USER"

# --- פונקציות עזר ---

def enhance_prompt(user_input, trigger_word):
    """שיפור הפרומפט בעזרת ג'מיני"""
    model = genai.GenerativeModel('gemini-1.5-flash')
    base_prompt = f"""
    Enhance this prompt for Flux AI to be hyper-realistic, 8k, cinematic.
    Subject trigger: "{trigger_word}".
    User request: "{user_input}"
    Output ONLY the English prompt.
    """
    try:
        response = model.generate_content(base_prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Error: {e}")
        return user_input # אם ג'מיני נכשל, נחזיר את המקור

# --- ה-End Points (מה שהעולם רואה) ---

@app.get("/")
def home():
    return {"status": "System is running", "service": "Hyper-Realism Generator"}

@app.post("/generate-image")
def generate_image(request: GenerationRequest):
    """מקבל בקשה, משפר עם ג'מיני, שולח ל-Fal"""
    try:
        # 1. שיפור הפרומפט
        final_prompt = enhance_prompt(request.prompt, request.trigger_word)
        print(f"Generating for: {final_prompt}")

        # 2. שליחה ל-Fal
        handler = fal_client.submit(
            "fal-ai/flux-lora",
            arguments={
                "prompt": final_prompt,
                "loras": [{"path": request.lora_url, "scale": 1.0}],
                "model_name": "flux-dev",
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "image_size": "landscape_4_3"
            },
        )
        
        # 3. קבלת התוצאה
        result = handler.get()
        return {
            "image_url": result['images'][0]['url'],
            "enhanced_prompt": final_prompt
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-user")
async def train_user(file: UploadFile = File(...), trigger_word: str = "OHAD_USER"):
    """
    מקבל קובץ ZIP עם תמונות, מעלה ל-FAL ומאמן מודל.
    מחזיר את ה-LORA URL שנשמור בבסיס הנתונים.
    """
    try:
        # שמירת הקובץ זמנית כדי להעלות אותו
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # העלאה לשרתים של Fal (כדי שיהיה URL נגיש לאימון)
        url = fal_client.upload_file(temp_filename)
        
        # ניקוי הקובץ הזמני
        os.remove(temp_filename)

        # התחלת אימון
        handler = fal_client.submit(
            "fal-ai/flux-lora-fast-training",
            arguments={
                "images_data_url": url,
                "trigger_phrase": trigger_word
            },
        )
        result = handler.get()
        return {"lora_url": result['diffusers_lora_file']['url']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
