from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("pipeline.pkl", "rb"))
df=pd.read_csv("cleaned.csv")


app=Flask(__name__)

@app.route('/')
def index():
    companies = sorted(df['company'].unique())
    car_model = sorted(df['name'].unique())
    kms_driven = sorted(df['kms_driven'].unique())
    fuel_type = sorted(df['fuel_type'].unique())
    year = sorted(df['year'].unique(), reverse=True)
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

@app.route('/compare')
def compare():
    companies = sorted(df['company'].unique())
    car_models = sorted(df['name'].unique())
    fuel_types = sorted(df['fuel_type'].unique())
    years = sorted(df['year'].unique(), reverse=True)

    # Map each company to its models
    company_model_map = {}
    for company in companies:
        models = sorted(df[df['company'] == company]['name'].unique())
        company_model_map[company] = models

    fuel_types.insert(0, "Select Type")
    years.insert(0, "Select Year")

    return render_template(
        'compare.html',
        companies=companies,
        car_models=car_models,
        fuel_types=fuel_types,
        years=years,
        company_model_map=company_model_map
    )

@app.route('/compare_result', methods=['POST'])
def compare_result():
    # Car 1 inputs
    company1 = request.form.get('company1')
    model1 = request.form.get('model1')
    fuel1 = request.form.get('fuel1')
    year1 = int(request.form.get('year1'))
    kms1 = int(request.form.get('kms1'))

    # Car 2 inputs
    company2 = request.form.get('company2')
    model2 = request.form.get('model2')
    fuel2 = request.form.get('fuel2')
    year2 = int(request.form.get('year2'))
    kms2 = int(request.form.get('kms2'))

    # Make predictions
    price1 = model.predict(pd.DataFrame([[model1, company1, year1, kms1, fuel1]],
                                        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))[0]
    price2 = model.predict(pd.DataFrame([[model2, company2, year2, kms2, fuel2]],
                                        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))[0]

    cheaper = "Car 1" if price1 < price2 else "Car 2"
    return render_template('compare_result.html',
                           price1=round(price1, 2),
                           price2=round(price2, 2),
                           cheaper=cheaper)


if __name__ == "__main__":
    app.run(debug=True)