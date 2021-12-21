from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import json

app = Flask(__name__)
model = tf.keras.models.load_model('.power.hdf5')

def data_split(input, reference=24):
    sequence = input.copy()
    X = []
    for i in range(reference, len(sequence)+1):
        X.append(np.array(sequence[i-reference: i]))
    return np.array(X)

@app.route("/test")
def test():
    return 'API is running'

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    data = json.loads(request.get_data())
    result = np.array([])
    delay = 48
    input = np.array(data)
    input = input.reshape(input.shape[0],1)

    for _ in range(delay):
        input_X = data_split(input)
        predicted_Y = model.predict(input_X)
        input = np.concatenate([input, predicted_Y])[-24:]
        result = np.append(result, predicted_Y)
    return json.dumps({'data': result.tolist()})

if __name__ == "__main__":
    app.run(debug=True)