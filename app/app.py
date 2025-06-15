from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from model.inference import inference
import os 
import base64
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('page.html')

@app.route('/health')
def health():
    return {'status': 'ok'}, 200

@app.route('/api/send', methods=['POST'])
def handle_upload():
    image = request.files.get('image')
    question = request.form.get('question')

    if not image or not question:
        return jsonify({"error" : "Missing image or question"})
    
    filename = secure_filename(image.filename)
    os.makedirs("uploads", exist_ok=True)
    temp_filename = os.path.join("uploads", filename)
    image.save(temp_filename)

    predicted_answer = inference(temp_filename, question)
    return jsonify({
        "message" : "Finished received data",
        "question" : question,
        "image_filename" : filename,
        "predicted_answer" : predicted_answer,
    })

# def receive_data():
#     data = request.get_json()
#     print("Received data:", data)
#     return jsonify({"status":"OK", "received": data})
    
# if __name__ == '__main__':
#     app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask on port: {port}")
    app.run(host='0.0.0.0', port=port, debug=False)