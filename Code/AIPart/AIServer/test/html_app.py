from flask import Flask, render_template

html_app = Flask(__name__)


@html_app.route('/')
def index():
    return render_template("TCP_test.html")


if __name__ == '__main__':
    html_app.run(debug=True, host='0.0.0.0', port=3356)
