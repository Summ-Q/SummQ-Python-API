import pymupdf 
from google import genai
from google.genai import types
import json
import os

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def generate_flashcards_from_chunk(text, images):
    """Function to generate flashcards from a small chunk of text/images."""
    prompt = """
    You are an educational expert. Read the following text and analyze the attached images (if any).
    Extract the most important concepts and formulate them into flashcards consisting of a question and an answer.
    
    CRITICAL INSTRUCTIONS:
    1. Use the images ONLY to extract information and understand the context. 
    2. DO NOT create questions that require the user to look at an image (e.g., avoid "What is shown in this figure?"), as the images will NOT be displayed to the user. All questions and answers must be entirely text-based and self-contained.
    3. Generate the Q&A in the same language as the provided text.
    
    The result must strictly be valid JSON format, as an array containing objects in the following structure:
    [
        {"question": "Question here", "answer": "Answer here"}
    ]
    Do not add any additional text or Markdown formatting outside the JSON.
    """
    
    contents_to_send = [prompt, f"Text Content:\n{text}\n"]
    
    if images:
        contents_to_send.append("Attached Images:")
        for img in images:
            contents_to_send.append(
                types.Part.from_bytes(data=img['bytes'], mime_type=f"image/{img['ext']}")
            )
            
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=contents_to_send
        )
        response_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        return json.loads(response_text)
    except Exception as e:
        print("Error parsing chunk:", e)
        return []

def process_pdf_for_flashcards(file_bytes):
    """Extracts text and images from a PDF and generates flashcards."""
    all_flashcards = []
    current_chunk_text = ""
    current_chunk_images = []
    PAGES_PER_CHUNK = 3 
    
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc):
            current_chunk_text += page.get_text() + "\n"
            
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                
                current_chunk_images.append({
                    "bytes": base_image["image"],
                    "ext": base_image["ext"]
                })
            
            if (page_num + 1) % PAGES_PER_CHUNK == 0:
                cards = generate_flashcards_from_chunk(current_chunk_text, current_chunk_images)
                all_flashcards.extend(cards)
                current_chunk_text = ""
                current_chunk_images = []
        
        # معالجة أي صفحات متبقية في آخر Chunk
        if current_chunk_text.strip() or current_chunk_images:
            cards = generate_flashcards_from_chunk(current_chunk_text, current_chunk_images)
            all_flashcards.extend(cards)
            
    return all_flashcards