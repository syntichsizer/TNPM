from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from serve import get_model_api

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model_api = get_model_api()


# default route

@app.route('/')
def index():
    return """
    <h1>TNPM api</h1>
    <p>Welcome. You can use this api on /api, by sending image url.</p>
    <p>Example:</p><a href="/api?image_url=https://images.dog.ceo/breeds/komondor/n02105505_1090.jpg">/api?image_url=https://images.dog.ceo/breeds/komondor/n02105505_1090.jpg</a>
    <p>Made by Karlo Sintić and Nino Nađ.</p><a href="https://github.com/karlosintic/TNPM">Contribute on Github.</a>
    """


# HTTP Errors handlers

@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


# API route

@app.route('/api', methods=['GET'])
def api():
    if 'image_url' in request.args:
        return model_api(str(request.args['image_url']))
    else:
        return "Error: No image url provided."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
