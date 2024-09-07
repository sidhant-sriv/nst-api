from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import uuid

# Import the style transfer function from styletransfer.py
from styletransfer import run_style_transfer_tflite

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/style-transfer", methods=["POST"])
def style_transfer():
    if "content_image" not in request.files or "style_image" not in request.files:
        return jsonify({"error": "Both content and style images are required"}), 400

    content_image = request.files["content_image"]
    style_image = request.files["style_image"]

    if content_image.filename == "" or style_image.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not (
        allowed_file(content_image.filename) and allowed_file(style_image.filename)
    ):
        return jsonify({"error": "File type not allowed"}), 400

    # Save uploaded files with secure filenames
    content_filename = secure_filename(content_image.filename)
    style_filename = secure_filename(style_image.filename)

    content_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_{content_filename}"
    )
    style_path = os.path.join(
        app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}_{style_filename}"
    )

    content_image.save(content_path)
    style_image.save(style_path)

    # Generate a unique output filename
    output_filename = f"{uuid.uuid4()}.jpg"

    # Run style transfer
    output_path = run_style_transfer_tflite(
        content_path,
        style_path,
        "path_to_tflite_model.tflite",  # Update this path
        output_filename,
    )

    # Clean up input files
    os.remove(content_path)
    os.remove(style_path)

    # Return the styled image
    return send_file(output_path, mimetype="image/jpeg")


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large"}), 413


if __name__ == "__main__":
    app.run(debug=False)

