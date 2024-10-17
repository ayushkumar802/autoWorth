from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("pipeline.pkl","rb"))
df=pd.read_csv("cleaned.csv")


app=Flask(__name__)

@app.route('/')
def index():
    companies = sorted(df['company'].unique())
    car_model = sorted(df['name'].unique())
    kms_driven = sorted(df['kms_driven'].unique())
    fuel_type = sorted(df['fuel_type'].unique())
    year = sorted(df['year'].unique(), reverse=True)
    companies.insert(0,"Select Company")
    fuel_type.insert(0, "Select Type")
    year.insert(0, "Select Year")

    return render_template('index.html', companies=companies, car_model=car_model,years=year, fuel_type=fuel_type, kms_driven=kms_driven)

@app.route('/predict', methods=['POST'])
def predict():
    company= request.form.get('company')
    car_model= request.form.get('car_model')
    year= int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame([[car_model, company,year, kms_driven,fuel_type]], columns=['name','company','year','kms_driven','fuel_type']))
    print(prediction)
    return str(np.round(prediction[0],2))

if __name__ == "__main__":
    app.run(debug=True)