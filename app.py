from flask import Flask, request, jsonify
import torch
from transformers import MusicgenForConditionalGeneration, AutoProcessor

app = Flask(__name__)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model and processor once when the app starts
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

@app.route("/generate", methods=["POST"])
def generate_music():
    try:
        # Get the text input from the request
        input_data = request.json
        text = input_data.get("text", "")

        # Process the input text
        inputs = processor(text=[text], padding=True, return_tensors="pt").to(device)

        # Generate audio values
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=128)

        # Convert tensor to list and return as JSON
        return jsonify({"generated_values": audio_values.cpu().tolist()})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
