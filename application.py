from flask import Flask, request, render_template

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PreditPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method =='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Sex = request.form.get('Sex'),
            Age = int(request.form.get('Age')),
            Job = int(request.form.get('Job')),
            Housing = request.form.get('Housing'),
            Saving_accounts = request.form.get('Saving_accounts'),
            Checking_account = request.form.get('Checking_account'),
            Credit_amount = int(request.form.get('Credit_amount')),
            Duration = int(request.form.get('Duration')),
            Purpose = request.form.get('Purpose')

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PreditPipeline()
        results = predict_pipeline.predict(pred_df)

        results = results[0]
        if results == 0.0:
            results = 'Not-Risky'
        else:
            results = 'Risky'

        return render_template('home.html', results = results)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0")
