from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle

# Load model
with open("obesitas_backend\svc_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Data dari front-end
    input_data = data["features"]
    prediction = model.predict([input_data])
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
