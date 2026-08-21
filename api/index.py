from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from ai_engine.card_generator import generate_flashcards_from_chunk, process_pdf_for_flashcards

app = FastAPI(title="Flashcards AI API")

@app.post("/generate-flashcards")
async def create_flashcards_endpoint(file: UploadFile = File(None), text: str = Form(None)):
    all_flashcards = []
    
    try:
        if file:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
            
            file_bytes = await file.read()
            # استدعاء دالة معالجة الـ PDF المعزولة
            all_flashcards = process_pdf_for_flashcards(file_bytes)
                
        elif text:
            if not text.strip():
                raise HTTPException(status_code=400, detail="No text found to generate flashcards.")
            # استدعاء دالة إنشاء البطاقات للنص المباشر
            all_flashcards = generate_flashcards_from_chunk(text, [])
            
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text.")
            
        if not all_flashcards:
            raise HTTPException(status_code=500, detail="AI model failed to generate flashcards from the provided content.")
            
        return {
            "status": "success",
            "message": "Flashcards extracted successfully",
            "data": all_flashcards
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")