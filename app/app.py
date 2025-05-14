from flask import Flask, request, jsonify
import joblib
from preprocess import preprocess_input

app = Flask(__name__)

model = joblib.load("../app/model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    X = preprocess_input(data)

    probas = model.predict_proba(X)[0]
    prob_yes = round(float(probas[1]), 2)
    
    prediction = "Yes" if prob_yes > 0.6 else "No"

    return jsonify({
        "prediction": prediction,
        "probability_of_yes": prob_yes
    })

if __name__ == "__main__":
    app.run(debug=True)
