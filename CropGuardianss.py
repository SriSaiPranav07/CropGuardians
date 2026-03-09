from flask import Flask, render_template, jsonify, request, redirect, url_for
from markupsafe import Markup
from model import predict_image
import utils
import pandas as pd

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Load CSV files
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')


# Home Page
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


# Prediction Route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            file = request.files['file']
            img = file.read()

            prediction = predict_image(img)
            print(prediction)

            res = Markup(utils.disease_dic[prediction])

            return render_template('display.html', status=200, result=res)

        except Exception as e:
            print(e)

    return redirect(url_for('home'))


# Shop Page
@app.route('/shop', methods=['GET', 'POST'])
def shop():
    return render_template(
        'shop.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )


# Run App
if __name__ == "__main__":
    app.run(debug=True)