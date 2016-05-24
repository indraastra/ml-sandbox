from base64 import b64decode
from collections import defaultdict
from io import BytesIO

from flask import Flask, jsonify, render_template, request
import numpy as np
from PIL import Image

from en_utils import image_to_numpy, load_glyph_set
from predict import load_classifier, predict_top_n
from imutils import image_windows


IMG_PREFIX = 'data:image/png;base64,'
IMG_SIZE = 20
IMG_WEIGHTS = 'weights/letters.mat'
GLYPH_SET = 'letters'
GLYPHS = load_glyph_set(GLYPH_SET)


app = Flask(__name__)
classifier = load_classifier(IMG_WEIGHTS)


@app.route('/')
def index():
    return render_template('draw.html')


@app.route('/classify', methods=['POST'])
def classify():
    results = {}
    if 'imageb64' not in request.form:
        return jsonify(results)

    data = request.form['imageb64']
    if not data.startswith(IMG_PREFIX):
        return jsonify(results)

    data = data[len(IMG_PREFIX):]
    image = Image.open(BytesIO(b64decode(data)))
    target_size = (IMG_SIZE, IMG_SIZE)

    all_scores = [0.0 for i in range(len(GLYPHS))]
    for subimage in image_windows(image, target_size,
                                  50, .75, .1):
        image = image_to_numpy(subimage)
        labels, scores = predict_top_n(classifier, image, limit=1)
        for label, score in zip(labels.flatten(), scores.flatten()):
            if score < .75: pass
            #all_scores[label] += score
            all_scores[label] = max(score, all_scores[label])

    results = []
    for label, score in enumerate(all_scores):
        results.append({
            'label': label,
            'score': score,
            'glyph': GLYPHS[label],
        })
    results.sort(key=lambda d: d['score'], reverse=True)

    print(results)
    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=True)

