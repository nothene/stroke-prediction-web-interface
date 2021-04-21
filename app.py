import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import json
import sys

cat_map = {}

with open('cat.json') as file:
    cat_map = json.load(file)

app = Flask(__name__)
model = pickle.load(open('gradient.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/table')
def table():
    return render_template('table.htm')

@app.route('/pic')
def pic():
    return render_template('pic.html')

@app.route('/predict', methods=['POST'])
def predict():
 
    int_features = []

    for i in request.form.values():
        if i in cat_map:
            int_features.append(int(cat_map[i]))
        else:
            int_features.append(float(i))

    final_features = [np.array(int_features)]

    prediction = model.predict(final_features)

    #return render_template('index.html', prediction=prediction)

    if prediction[0] == 1:
        return render_template('index.html', prediction="You have stroke!")
    else:
        return render_template('index.html', prediction="You don't have stroke, for now")

    

if __name__ == "__main__":
    app.run(debug=True)