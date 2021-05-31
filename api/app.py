# API example

from flask import Flask, jsonify, request
from flask_cors import CORS
import logging_mlflow as lm
import os
from collections import Counter

app = Flask(__name__)
CORS(app)
r = lm.Repositorio(os.environ['MLFLOW_URI'])
current_version = None
model = None

@app.route('/<model_name>/predict', methods=['POST'])
def get_predict(model_name):
    global current_version
    global model

    json = request.get_json()

    sentences = []

    for sentence in json['data']:
        sentences.append(sentence['text'])

    model_info = r.get_production_version(model_name)

    if current_version != model_info['version']:
        model = r.get_model(model_name, 'sklearn')
        current_version = model_info['version']

    result = model.predict(sentences)
    result_dict = dict(Counter(result))

    info = {
        "model": model_name,
        "version": current_version,
    }

    return_dict = {
        "results": result_dict,
        "info": info,
    }

    return return_dict


if __name__=="__main__":
    app.run(host='0.0.0.0', port=5005)