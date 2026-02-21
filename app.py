import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables securely
load_dotenv()

app = Flask(__name__)

# Initialize the new Google GenAI client
client = genai.Client()
MODEL_ID = "gemini-2.5-flash"

# Define the persona
EDUGENIE_PERSONA = """You are EduGenie – a friendly, intelligent, and student-focused AI Learning Assistant powered by Google Gemini, designed to make learning simple, engaging, and stress-free; your role is to act like a patient teacher and motivating mentor who explains concepts clearly in simple language, breaks down complex topics into easy step-by-step explanations, uses bullet points when helpful, provides relatable examples, avoids unnecessary technical jargon, and encourages curiosity and confidence.

When answering questions, give structured and clean responses.
When generating quizzes, create engaging multiple-choice questions with a mix of easy, medium, and slightly challenging levels.
When summarizing study material, convert long content into short, crisp, exam-friendly bullet points highlighting key ideas.

Maintain a positive, supportive tone throughout and end ALL responses with this exact short motivational line: "Keep learning, you're doing great! 🚀" """

# Standard config for Ask and Summarize
genai_config = types.GenerateContentConfig(
    system_instruction=EDUGENIE_PERSONA,
)

# STRICT JSON config specifically for the Interactive Quiz
quiz_config = types.GenerateContentConfig(
    system_instruction=EDUGENIE_PERSONA,
    response_mime_type="application/json",
)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"Student Question: {question}",
            config=genai_config
        )
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/quiz", methods=["POST"])
def quiz():
    data = request.get_json()
    topic = data.get("topic")
    if not topic:
        return jsonify({"error": "Topic cannot be empty"}), 400

    # We specifically instruct the model on the exact JSON structure we want
    prompt = f"""Generate a 5-question multiple-choice quiz on this topic: {topic}. 
    You MUST return ONLY a JSON array of objects. 
    Each object must have exactly these keys:
    - "question": The question text.
    - "options": An array of exactly 4 string choices.
    - "answer": The exact string of the correct choice from the options array."""

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=quiz_config
        )
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/summary", methods=["POST"])
def summary():
    data = request.get_json()
    material = data.get("material")
    if not material:
        return jsonify({"error": "Study material cannot be empty"}), 400

    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=f"Please summarize the following study material:\n\n{material}",
            config=genai_config
        )
        return jsonify({"result": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)