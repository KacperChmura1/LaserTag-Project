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
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#Loading data functions
def load_data():
    df = pd.read_csv("Stats.csv",index_col = 0)
    df["Accuracy"] = df["Accuracy"].apply(proc).apply(float)
    return df
def proc(acc):
    acc = acc[:-1]
    return acc
#Loading model function    
def load_model(model_name):
    with open (model_name, "rb") as file:
        data = pickle.load(file)
    return data
#Header
def header(url):
     st.subheader(f'<p style="background-color:#0066cc;color:#33ff33;font-size:24px;border-radius:2%;">{url}</p>')
df = load_data()
df_X = df.drop("Rating", axis = 1)
#Loading Data to Check Model(avoid data leakage)
X_test = pd.read_csv("X_test.csv",index_col = 0)
X_test_3 = pd.read_csv("X_test_3.csv",index_col = 0)
y_test = pd.read_csv("y_test.csv",index_col = 0)
y_mean = np.mean(y_test)

model_choose = "model"
#Predict Page

def show_predict_page():
    st.title("Rating Predictions")
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> Model predict well if rating is bellow 100!</p>', unsafe_allow_html=True)

    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> I need some information to predict your rating:', unsafe_allow_html=True)
    #Stats from data  
    shot_fired = st.number_input("Shot fired",step = 1)
    hits = st.number_input("Hits",step = 1)
    deaths = st.number_input("Deaths",step = 1)
    dmg_get = st.number_input("DMG Get",step = 1)
    #Select box with models
    cheated = False
    cheated = st.checkbox("Cheated Model")
    if cheated:
        cheat_string = "_3.pkl"
    else:
        cheat_string = ".pkl"
    
           
    model_choose = st.sidebar.selectbox("Model choose", {"Gradient Boost", "Linear Regression","Forest","KNN","SVR Linear"})
    if model_choose == "Gradient Boost":
        model_name = "Models/gradient"+cheat_string
        model = load_model(model_name)
    elif model_choose == "Linear Regression":
        model_name = "Models/linear_regression"+cheat_string
        model = load_model(model_name)
    elif model_choose == "Forest":
        model_name = "Models/forest"+cheat_string
        model = load_model(model_name)
    elif model_choose == "KNN":
        model_name = "Models/KNN"+cheat_string
        model = load_model(model_name)
    elif model_choose == "SVR Linear":
        model_name = "Models/SVRLinear"+cheat_string
        model = load_model(model_name)

    ok = st.button("Calculate Rating")
    #Avoid div by 0
    if ok:
        if shot_fired == 0:
            accuracy = 0
        else:
            accuracy = (hits / shot_fired)*100
        #Predicting Inputed Vale
        if cheat_string == ".pkl":
            X = np.array([[accuracy,shot_fired,hits,deaths, dmg_get]])
        else:
            X = np.array([[accuracy,hits, dmg_get]])
        rating = model.predict(X)
        real_rating = (hits/(dmg_get + 1) * (accuracy*2+100))/3
        
        prediction = f'<p style="font-family:Courier; color:Red; font-size: 30px;">Model Predict: {rating[0]:.2f}</p>'
        true_value = f'<p style="font-family:Courier; color:Green; font-size: 30px;">Real Rating: {real_rating:.2f}</p>'
        #Display Results
        st.markdown(prediction, unsafe_allow_html=True)
        st.markdown(true_value, unsafe_allow_html=True)  
        #Predict Values from Loaded Data
        if cheat_string == ".pkl":
            ratings = model.predict(X_test)
        else:
            ratings = model.predict(X_test_3)
        
        st.title("How good is our model")
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">For best results our model was tested with multiple hyper parameters with GridSearchCV.</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">Here are the results:</p>', unsafe_allow_html=True)
        #How Good is Model?
        mae = metrics.mean_absolute_error(y_test, ratings)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, ratings))
        error_y_mean = (mae/np.mean(y_test))*100
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">MAE: {mae:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">RMSE: {rmse:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">Error in %: {error_y_mean[0]:.2f}%</p>', unsafe_allow_html=True)

        #Display Results Plot with Plotly
        y_test_list = y_test.values.tolist()
        ratings_list = ratings.tolist()
        y_test_unpacked = []
        for c in y_test_list:
            for t in c:
                y_test_unpacked.append(t)
        
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig3.add_trace(
            go.Scatter(x=y_test_unpacked, y = np.round(ratings_list,2), name="Predictions",mode = "markers",
            marker=dict(
            color='rgba(135, 206, 250, 0.6)',
            size=8
            )
        ),
        ),
        fig3.add_trace(
            go.Scatter(x=y_test_unpacked, y = y_test_unpacked, name="Real Ratings"),
            secondary_y=True,
        )
        # Add figure title
        fig3.update_layout(
        title_text=f" {model_choose}: Predictions vs True Values"
        )
        st.plotly_chart(fig3)
        #Add model_3 to app
        #Describe project in explore page!!!!
        

        

    
