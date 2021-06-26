# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 21:36:26 2021

@author: SHANMATHU
"""

from flask import Flask, render_template, request
import pickle
app = Flask(__name__)
capacity_model=pickle.load(open("lin_poly_5.pkl" ,'rb'))
cycle_model=pickle.load(open("svr_5.pkl",'rb'))
st=pickle.load(open("Minmaxscaler1_5.pkl",'rb'))
st2=pickle.load(open("Minmaxscaler2_5.pkl",'rb'))
pr = pickle.load(open("poly_5.pkl",'rb'))

@app.route("/")
def home():
    return render_template('new.html')

@app.route("/predictor", methods=['GET', 'POST'])
def predictor():
    return render_template('basic.html')

@app.route("/predict", methods = ['POST', 'GET'])
def predict():
        dis_max_time=request.form['dis_max_time']
        dis_min_vol_time=request.form["dis_min_vol_time"]
        ch_max_vol_time=request.form["ch_max_vol_time"]
        capacity=capacity_model.predict(pr.transform(st.transform([[ch_max_vol_time,dis_min_vol_time]])))
        cycle=cycle_model.predict(st2.transform([[capacity,ch_max_vol_time,dis_max_time]]))
        return render_template('predict.html', 
                       capacity=round(capacity[0],3),
                       cycle=round(cycle[0]))

if __name__ == '__main__':
	app.run(debug = True)