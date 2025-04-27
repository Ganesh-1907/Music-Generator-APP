from flask import Flask, request, jsonify, send_file, render_template
from transformers import pipeline
import scipy.io.wavfile as wav
import numpy as np
import os

app = Flask(__name__)

# Load MusicGen model
synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_audio():
    try:
        data = request.json
        text_prompt = data.get("text")

        if not text_prompt:
            return jsonify({"error": "No text provided"}), 400

        # Generate music
        music = synthesiser(text_prompt, forward_params={"do_sample": True})
        print(music,'Music Ganesh')

        if "audio" in music and len(music["audio"]) > 0:
            # Save as WAV file
            file_path = "musicgen_out.wav"
            wav.write(file_path, rate=music["sampling_rate"], data=np.array(music["audio"]))

            return jsonify({"message": "Audio generated successfully!", "audio_url": file_path})

        return jsonify({"error": "Failed to generate audio"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download")
def download_audio():
    file_path = "musicgen_out.wav"
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
