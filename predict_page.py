from ctypes.wintypes import HINSTANCE
from pickletools import StackObject
from pyexpat import model
from statistics import mode
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import tensorflow as tf

def load_data():
    df = pd.read_csv("Stats.csv",index_col = 0)
    df["Accuracy"] = df["Accuracy"].apply(proc).apply(float)
    return df
#I need to load separate data to avoid data leakage
    
def proc(acc):
    acc = acc[:-1]
    return acc
  
def load_gradient_boost():
    with open ("gradient.pkl", "rb") as file:
        data = pickle.load(file)
    return data
def linear_regression():
    with open ("linear_regression.pkl", "rb") as file:
        data = pickle.load(file)
    return data
def forest():
    with open ("forest.pkl", "rb") as file:
        data = pickle.load(file)
    return data
def KNN():
    with open ("KNN.pkl", "rb") as file:
        data = pickle.load(file)
    return data

    #with open ("forest.pkl", "rb") as file:
    #    data = pickle.load(file)
    #return data

def header(url):
     st.subheader(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>')
df = load_data()
df_X = df.drop("Rating", axis = 1)

#f_X = df[["Accuracy","Shot fired","Hits","Deaths","DMG_Get"]]
X_test = pd.read_csv("X_test.csv",index_col = 0)
y_test = pd.read_csv("y_test.csv",index_col = 0)
y_mean = np.mean(y_test)
def show_predict_page():
    st.title("Rating Predictions")
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> Model predict well if rating is bellow 100!</p>', unsafe_allow_html=True)

    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> I need some information to predict your rating:', unsafe_allow_html=True)
    


    shot_fired = st.number_input("Shot fired",step = 1)
    hits = st.number_input("Hits",step = 1)
    deaths = st.number_input("Deaths",step = 1)
    dmg_get = st.number_input("DMG Get",step = 1)
    

    model_choose = st.sidebar.selectbox("Model choose", {"Gradient Boost", "Linear Regression","Forest","KNN"})
    if model_choose == "Gradient Boost":
        model = load_gradient_boost()
    elif model_choose == "Linear Regression":
        model = linear_regression()
    elif model_choose == "Forest":
        model = forest()
    elif model_choose == "KNN":
        model = KNN()
    elif model_choose == "Linear Regression":
        model = linear_regression()

    ok = st.button("Calculate Rating")
    if ok:
        if shot_fired == 0:
            accuracy = 0
        else:
            accuracy = (hits / shot_fired)*100
        
        X = np.array([[accuracy,shot_fired,hits,deaths, dmg_get]])
        rating = model.predict(X)
        real_rating = (hits/(dmg_get + 1) * (accuracy*2+100))/3
        prediction = f'<p style="font-family:Courier; color:Red; font-size: 30px;">Model Predict: {rating[0]:.2f}</p>'
        true_value = f'<p style="font-family:Courier; color:Green; font-size: 30px;">Real Rating: {real_rating:.2f}</p>'

        st.markdown(prediction, unsafe_allow_html=True)
        st.markdown(true_value, unsafe_allow_html=True)  

        ratings = model.predict(X_test)
        st.title("How good is our model")
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">For best results our model was tested with multiple hyper parameters with GridSearchCV.</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">Here are the results:</p>', unsafe_allow_html=True)

        mae = metrics.mean_absolute_error(y_test, ratings)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, ratings))
        error_y_mean = (mae/np.mean(y_test))*100
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">MAE: {mae:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">RMSE: {rmse:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">Error in %: {error_y_mean[0]:.2f}%</p>', unsafe_allow_html=True)

        fig = plt.figure()
        scater = plt.scatter(y_test,ratings)
         #Perfect predictions
        line = plt.plot(y_test,y_test,'r')
        st.title("The effectiveness of our model")
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">The red line is the correct answer and the blue point is our prediction</p>', unsafe_allow_html=True)
        st.pyplot(fig)
       

    
