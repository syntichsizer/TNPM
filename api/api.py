from flask import Flask, request, jsonify
from flask_cors import CORS
from serve import get_model_api

app = Flask(__name__)
model_api = get_model_api()


# default route
@app.route('/')
def index():
    return """
    <h1>TNPM api</h1>
    <p>Welcome. You can use this api on /api, by sending image url.</p>
    <p>Example:</p><a href="/api?img=https://images.dog.ceo/breeds/komondor/n02105505_1090.jpg">/api?img=https://images.dog.ceo/breeds/komondor/n02105505_1090.jpg</a>
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

    if 'img' in request.args:
        return model_api(str(request.args['img']))
    else:
        return "Error: No image url provided."


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
