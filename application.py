import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

#import ridge regression and standard scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Status=float(request.form.get('Status'))
        AdultMortality = float(request.form.get('Adult Mortality'))
        infantdeaths = float(request.form.get('infant deaths'))
        Alcohol = float(request.form.get('Alcohol'))
        percentageexpenditure = float(request.form.get('percentage expenditure'))
        HepatitisB = float(request.form.get('Hepatitis B'))
        Measles = float(request.form.get('Measles'))
        BMI = float(request.form.get('BMI'))
        Polio = float(request.form.get('Polio'))
        Totalexpenditure = float(request.form.get('Total expenditure'))
        Diphtheria  = float(request.form.get('Diphtheria'))
        HIV_AIDS = float(request.form.get('HIV/AIDS'))
        Population = float(request.form.get('Population'))
        thinness1_19years = float(request.form.get('thinness  1-19 years'))
        Incomecompositionofresources = float(request.form.get('Income composition of resources'))
        Schooling = float(request.form.get('Schooling'))

        new_data_scaled=standard_scaler.transform([[Status,AdultMortality,infantdeaths,Alcohol,percentageexpenditure,HepatitisB,Measles,BMI,Polio,Totalexpenditure,Diphtheria,HIV_AIDS,Population,thinness1_19years,Incomecompositionofresources,Schooling]])
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)
