from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

app = Flask(__name__)

# Load tokenizer and model from Hugging Face Hub
HF_REPO = "Ahmedg120/t5-question-gen"

tokenizer = T5Tokenizer.from_pretrained(HF_REPO)
model = T5ForConditionalGeneration.from_pretrained(HF_REPO)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to generate questions
def generate_questions(description, num_questions=10):
    input_text = f"Given the following job description, generate a specific technical interview question: {description}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=num_questions,
        early_stopping=True
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# POST endpoint to generate questions
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or "description" not in data:
        return jsonify({"error": "Missing 'description' in request body."}), 400
    try:
        questions = generate_questions(data["description"])
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# GET endpoint for health check
@app.route('/', methods=['GET'])
def home():
    return "T5 Interview Question Generator is running!"

# Run the app locally
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
