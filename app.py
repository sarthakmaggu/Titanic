#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:40:33 2020

@author: maggu
"""

import pandas as pd
from flask import Flask, jsonify, request
import pickle
import json

model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)
    result = model.predict(data_df)
    output = {'results': int(result[0])}
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)


