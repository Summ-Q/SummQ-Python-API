from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pymupdf 
from google import genai 
import json
import os 

# قراءة الـ API Key من بيئة التشغيل للأمان
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI(title="Flashcards AI API")

def generate_flashcards(text):
    """Function to send text to the LLM and receive Q&A as JSON."""
    prompt = f"""
    You are an educational expert. Read the following text and extract the most important concepts.
    Formulate them into flashcards consisting of a question and an answer.
    Generate the Q&A in the same language as the provided text.
    
    The result must strictly be valid JSON format, as an array containing objects in the following structure:
    [
        {{"question": "Question here", "answer": "Answer here"}}
    ]
    Do not add any additional text or Markdown formatting outside the JSON.
    
    Text:
    {text}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt
        )
        
        response_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        flashcards = json.loads(response_text)
        return flashcards
    except Exception as e:
        print("Error parsing the JSON output:", e)
        return None

# التعديل هنا: خلينا الـ endpoint يقبل (file) أو (text)
@app.post("/generate-flashcards/")
async def create_flashcards_endpoint(
    file: UploadFile = File(None), 
    text: str = Form(None)
):
    extracted_text = ""
    
    try:
        # 1. لو اليوزر باعت ملف PDF
        if file:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
            
            file_bytes = await file.read()
            with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    extracted_text += page.get_text()
                    
        # 2. لو اليوزر باعت نص مباشر
        elif text:
            extracted_text = text
            
        # 3. لو مبعتش ولا ده ولا ده
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text.")
            
        # التأكد إن في كلام فعلاً عشان نبعته للموديل
        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text found to generate flashcards.")
            
        # توليد الفلاش كاردز
        cards = generate_flashcards(extracted_text)
        
        if not cards:
            raise HTTPException(status_code=500, detail="AI model failed to generate flashcards.")
            
        return {
            "status": "success",
            "message": "Flashcards generated successfully",
            "data": cards
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")