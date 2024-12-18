import os
import tempfile
import time
from flask import Flask, request, render_template, redirect, session, url_for
from transformers import pipeline
import PyPDF2
import fitz  # PyMuPDF
import camelot  # Table extraction
import pytesseract
from PIL import Image
import concurrent.futures
import requests
import random
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
import uuid

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Random secret key generation (you can change this to use your own random key or dotenv for secure key management)
app.secret_key = os.getenv("SECRET_KEY", str(uuid.uuid4()))  # Fallback to a randomly generated UUID if not found in .env

# Gemini API credentials
GEMINI_API_KEY = "AIzaSyBSko2EMrZ6Oe15NsgnY5NB1jHeINQf1w0"  # Replace with your API key
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent" # Fallback if not set

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

# Summarization pipeline using T5
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Path to Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Global variable to store extracted PDF content
document_text = ""


# ------------------ Helper Functions ------------------ #

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def pdf_to_text(file_path):
    """Extract text from PDF using PyPDF2."""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text


def summarize_text(text, chunk_size=300):
    """Summarize large text by splitting into chunks."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_chunk = {
            executor.submit(summarizer, chunk, max_length=100, min_length=20): chunk
            for chunk in chunks
        }
        for future in concurrent.futures.as_completed(future_to_chunk):
            try:
                summary = future.result()[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                summaries.append(f"Error: {e}")

    return " ".join(summaries)


def extract_tables_from_pdf(pdf_path):
    """Extract tables from PDF using Camelot."""
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        table_summaries = []

        for table in tables:
            table_data = table.df.to_string(index=False, header=False)
            summary = summarize_text(table_data)
            table_summaries.append(summary)

        return table_summaries
    except Exception:
        return ["Error extracting tables."]


def extract_images_from_pdf(pdf_path, output_folder="static/images"):
    """Extract images from PDF using PyMuPDF."""
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []

    for page_number in range(len(doc)):
        for img_index, img in enumerate(doc[page_number].get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_path = os.path.join(output_folder, f"page_{page_number+1}_img_{img_index+1}.{image_ext}")
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)
            image_paths.append(image_path)
    return image_paths


def summarize_text_from_images(image_paths):
    """Summarize text extracted from images using Tesseract OCR."""
    ocr_texts = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(pytesseract.image_to_string, Image.open(img)): img
                           for img in image_paths}
        for future in concurrent.futures.as_completed(future_to_image):
            try:
                ocr_texts.append(future.result())
            except Exception:
                ocr_texts.append("")

    combined_text = " ".join(ocr_texts)
    return summarize_text(combined_text) if combined_text else "No text found in images."


def ask_gemini(question, document_text):
    """Generate answers using Gemini API."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": f"Document Context: {document_text}\n\nQuestion: {question}"}]}]
    }
    try:
        response = requests.post(
            GEMINI_ENDPOINT,
            headers=headers,
            json=payload,
            params={"key": GEMINI_API_KEY}
        )
        response_json = response.json()
        answer = response_json['candidates'][0]['content']['parts'][0]['text']
        return answer or "No answer generated."
    except Exception as e:
        return f"Error: {str(e)}"


def generate_quiz_questions_via_gemini(text, num_questions=10):  # Set to 10
    headers = {"Content-Type": "application/json"}
    prompt = (
        f"Generate {num_questions} multiple-choice questions (MCQs) based on the following text:\n"
        f"{text}\n"
        f"Each question should have:\n"
        "- A single question statement\n"
        "- Exactly 4 options labeled (A), (B), (C), and (D)\n"
        "- Mention the correct answer clearly.\n\n"
        "Format the response like this:\n"
        "Q1: <Question>\n(A) Option1\n(B) Option2\n(C) Option3\n(D) Option4\nAnswer: OptionLabel"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        response = requests.post(
            GEMINI_ENDPOINT,
            headers=headers,
            json=payload,
            params={"key": GEMINI_API_KEY}
        )

        if response.status_code == 200:
            response_json = response.json()
            questions_text = response_json['candidates'][0]['content']['parts'][0]['text']

            questions = []
            for question_block in questions_text.split('\n\n'):
                question_data = {}
                lines = question_block.split('\n')
                if len(lines) >= 5:
                    question_data['question'] = lines[0].strip()
                    options = {chr(65 + i): lines[i+1].strip() for i in range(4)}
                    question_data['options'] = options
                    question_data['answer'] = lines[5].split(':')[1].strip().replace('(', '').replace(')', '')
                    questions.append(question_data)

            return questions

        else:
            return [{"question": "Error generating questions.", "options": {}, "answer": "Error"}]

    except Exception as e:
        print(f"Error generating questions: {str(e)}")
        return [{"question": "Error generating questions.", "options": {}, "answer": "Error"}]

# ------------------ Routes ------------------ #

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global document_text
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        file_path = tmp_file.name
        file.save(file_path)

    document_text = pdf_to_text(file_path)
    text_summary = summarize_text(document_text)
    table_summaries = extract_tables_from_pdf(file_path)
    image_paths = extract_images_from_pdf(file_path)
    image_summary = summarize_text_from_images(image_paths)

    combined_summary = f"Text Summary:\n{text_summary}\n\nTable Summaries:\n{' '.join(table_summaries)}\n\nImage Summary:\n{image_summary}"
    os.remove(file_path)

    return render_template('result.html', summary=combined_summary, image_paths=image_paths)


@app.route('/ask', methods=['GET', 'POST'])
def ask_question():
    global document_text
    if request.method == 'POST':
        question = request.form.get('question')
        answer = ask_gemini(question, document_text)
        return render_template('qa.html', question=question, answers=[answer])
    return render_template('qa.html')


@app.route('/quiz', methods=['GET', 'POST'])
def quiz_page():
    global document_text

    # GET Request: Generate or display quiz questions
    if request.method == 'GET':
        if 'quiz_questions' not in session or not session['quiz_questions']:
            session['quiz_questions'] = generate_quiz_questions_via_gemini(document_text)

        if not session['quiz_questions'] or isinstance(session['quiz_questions'], str):
            return render_template('quiz.html', error_message="No quiz questions generated. Please try uploading a document again.")

        session['user_answers'] = [None] * len(session['quiz_questions'])
        return render_template('quiz.html', questions=session['quiz_questions'])

    # POST Request: Capture answers from the quiz
    if request.method == 'POST':
        answers = {}
        
        # Capture answers for each question
        for i, question in enumerate(session['quiz_questions']):
            question_name = f'question_{i}'
            answers[question_name] = request.form.get(question_name)  # Get the selected answer

        correct_answers = 0
        feedback = []

        # Compare user answers with correct answers
        for i, q in enumerate(session['quiz_questions']):
            user_answer = answers.get(f'question_{i}')
            is_correct = user_answer == q['answer']
            if is_correct:
                correct_answers += 1

            # Collect feedback for each question
            feedback.append({
                'question': q['question'],
                'your_answer': user_answer,
                'correct_answer': q['answer'],
                'is_correct': is_correct,
                'suggestion': q['options'].get(user_answer, "Review the question.")
            })

        overall_feedback = "Good job!" if correct_answers >= len(session['quiz_questions']) // 2 else "Keep practicing!"

        return render_template('quiz_result.html', score=correct_answers, total=len(session['quiz_questions']), feedback=feedback, overall_feedback=overall_feedback)



if __name__ == '__main__':
    app.run(debug=True)
