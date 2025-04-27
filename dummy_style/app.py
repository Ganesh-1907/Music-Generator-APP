from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/data.json')
def data():
    return send_from_directory('.', 'data.json')

if __name__ == '__main__':
    app.run(debug=True)
