from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pymupdf 
from google import genai
from google.genai import types # ضفنا دي عشان نحدد نوع البيانات (صور) للموديل
import json
import os
import base64 # ضفنا دي عشان تحويل الصور

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

app = FastAPI(title="Flashcards AI API")

def generate_flashcards(text, images):
    """Function to send text and images to the LLM and receive Q&A as JSON."""
    prompt = """
    You are an educational expert. Read the following text and analyze the attached images (if any).
    Extract the most important concepts and formulate them into flashcards consisting of a question and an answer.
    
    CRITICAL INSTRUCTION: If a question heavily relies on or refers to a specific image, you MUST include the provided Image ID in the 'image' field. If no image is needed for the question, set the 'image' field to null.
    Generate the Q&A in the same language as the provided text.
    
    The result must strictly be valid JSON format, as an array containing objects in the following structure:
    [
        {"question": "Question here", "answer": "Answer here", "image": "image_id_or_null"}
    ]
    Do not add any additional text or Markdown formatting outside the JSON.
    """
    
    # تجهيز محتوى الرسالة (النص الأساسي)
    contents_to_send = [prompt, f"Text Content:\n{text}\n"]
    
    # لو في صور، هنضيفها للرسالة ونعطي لكل صورة ID عشان الموديل يعرف يربطها
    if images:
        contents_to_send.append("Attached Images:")
        for img in images:
            contents_to_send.append(f"\n[Image ID: {img['id']}]")
            # بنجهز الصورة بصيغة يقدر الموديل يفهمها
            contents_to_send.append(
                types.Part.from_bytes(
                    data=img['bytes'],
                    mime_type=f"image/{img['ext']}"
                )
            )
            
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=contents_to_send
        )
        
        response_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        return json.loads(response_text)
    except Exception as e:
        print("Error parsing the JSON output:", e)
        return None

@app.post("/generate-flashcards/")
async def create_flashcards_endpoint(file: UploadFile = File(None), text: str = Form(None)):
    extracted_text = ""
    extracted_images = [] # ليست هنخزن فيها بيانات الصور
    
    try:
        if file:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(status_code=400, detail="Invalid file format. Please upload a PDF.")
            
            file_bytes = await file.read()
            image_counter = 1 # عداد عشان ندي لكل صورة رقم مختلف
            
            with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    extracted_text += page.get_text()
                    
                    # الكود الجديد لاستخراج الصور من كل صفحة
                    for img_info in page.get_images(full=True):
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"] # الصورة كـ Bytes (عشان نبعتها للموديل)
                        image_ext = base_image["ext"]     # امتداد الصورة (png, jpeg, etc.)
                        
                        # تحويل الصورة لـ Base64 عشان نبعتها لبتاع الـ Laravel كـ Text
                        b64_string = base64.b64encode(image_bytes).decode("utf-8")
                        img_id = f"image_{image_counter}"
                        
                        extracted_images.append({
                            "id": img_id,
                            "bytes": image_bytes,
                            "ext": image_ext,
                            "base64": b64_string
                        })
                        image_counter += 1
                        
        elif text:
            extracted_text = text
            
        else:
            raise HTTPException(status_code=400, detail="Please provide either a PDF file or text.")
            
        if not extracted_text.strip() and not extracted_images:
            raise HTTPException(status_code=400, detail="No content found to generate flashcards.")
            
        # نبعت النص والصور للموديل
        cards = generate_flashcards(extracted_text, extracted_images)
        
        if not cards:
            raise HTTPException(status_code=500, detail="AI model failed to generate flashcards.")
            
        # تجهيز قاموس (Dictionary) فيه الصور كـ Base64 للـ Laravel
        images_for_laravel = {img["id"]: img["base64"] for img in extracted_images}
        
        return {
            "status": "success",
            "message": "Flashcards and images extracted successfully",
            "data": {
                "flashcards": cards,
                "images_base64": images_for_laravel
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")