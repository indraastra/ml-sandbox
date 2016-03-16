from base64 import b64decode
from io import BytesIO

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image

from en_utils import image_to_numpy
from predict import load_classifier

IMG_PREFIX = "data:image/png;base64,"
IMG_SIZE = 20
IMG_WEIGHTS = 'weights/ex4weights_new.mat'

app = Flask(__name__)
classifier = load_classifier(IMG_WEIGHTS)


@app.route('/')
def index():
    return render_template('draw.html')


@app.route('/classify', methods=['POST'])
def classify():
    results = {'results': []}
    if 'imageb64' not in request.form:
        return jsonify(results)

    data = request.form['imageb64']
    if not data.startswith(IMG_PREFIX):
        return jsonify(results)

    data = data[len(IMG_PREFIX):]
    image = Image.open(BytesIO(b64decode(data)))
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = image_to_numpy(image)
    print(image.reshape(IMG_SIZE, IMG_SIZE, order='F').astype(np.uint8))
    res = classifier(image).tolist()
    print(res)
    results['results'] = res
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True)

