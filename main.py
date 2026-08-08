import pymupdf 
from google import genai 
import json
import os # تأكد من إضافة المكتبة دي

# كده الكود هيقرأ المفتاح من النظام ومش هيكون مكشوف لأي حد
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI(title="Flashcards AI API")
# ... باقي الكود زي ما هو ...
def generate_flashcards(text):
    """Function to send text to the LLM and receive Q&A as JSON."""
    model = genai.GenerativeModel('gemini-3.6-flash')
    
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
    
    response = model.generate_content(prompt)
    
    try:
        response_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        flashcards = json.loads(response_text)
        return flashcards
    except Exception as e:
        print("Error parsing the JSON output:", e)
        return None

# 3. Create the API Endpoint
@app.post("/generate-flashcards/")
async def create_flashcards_endpoint(file: UploadFile = File(...)):
    """
    This endpoint receives a PDF file, extracts text, 
    and returns generated flashcards in JSON format.
    """
    # Check if the uploaded file is a PDF
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
    
    try:
        # Read the file content into memory
        file_bytes = await file.read()
        text = ""
        
        # Open the PDF directly from the memory stream (no need to save it locally)
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
                
        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract any text from the provided PDF.")
            
        # Generate the flashcards
        cards = generate_flashcards(text)
        
        if not cards:
            raise HTTPException(status_code=500, detail="AI model failed to generate flashcards.")
            
        # Return the success response
        return {
            "status": "success",
            "message": "Flashcards generated successfully",
            "data": cards
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")