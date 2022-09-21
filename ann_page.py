from unicodedata import decimal
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
#Loading Data
X_test = pd.read_csv("data/X_test.csv",index_col = 0)
X_train = pd.read_csv("data/X_train_ann.csv",index_col = 0)
y_train = pd.read_csv("data/y_train_ann.csv",index_col = 0)
y_test = pd.read_csv("data/y_test.csv",index_col = 0)
#Scaling Data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
#Loading model
def load_model():
    model = tf.keras.models.load_model("models/ann.hdf5")
    return model
#Whole ANN page
def show_ann_page():
    ann_model = load_model()
    st.title("Rating Predictions")
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> Model predict well if rating is bellow 100!</p>', unsafe_allow_html=True)

    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> I need some information to predict your rating:', unsafe_allow_html=True)
    #Stats from data  
    shot_fired = st.number_input("Shot fired",step = 1)
    hits = st.number_input("Hits",step = 1)
    deaths = st.number_input("Deaths",step = 1)
    dmg_get = st.number_input("DMG Get",step = 1)
    ok = st.button("Calculate Rating")
    #Avoid div by 0
    accuracy = 0
    if ok:
        if shot_fired == 0:
            accuracy = 0
        else:
            accuracy = (hits / shot_fired)*100
            
        X = np.array([[accuracy,shot_fired,hits,deaths, dmg_get]])
        #Predicting Inputed Vale
        X = scaler.transform(X)
        rating = ann_model.predict(X)
        real_rating = (hits/(dmg_get + 1) * (accuracy*2+100))/3
        real_rating = np.round(real_rating,2)
        #Prediction vs Real Rating
        prediction = f'<p style="font-family:Courier; color:Red; font-size: 30px;">Model Predict: {np.round(rating[0,0])}</p>'
        true_value = f'<p style="font-family:Courier; color:Green; font-size: 30px;">Real Rating: {real_rating}</p>'
        st.markdown(prediction, unsafe_allow_html=True)
        st.markdown(true_value, unsafe_allow_html=True)
        #How Good is Model?
        st.title("How good is our model")
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">For best results our model was tested with multiple layer layouts.</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">Here are the results:</p>', unsafe_allow_html=True)
        
        ratings = ann_model.predict(X_test_s)
        #st.markdown(y_test)
        #st.markdown(ratings)
        ratings_list = ratings.tolist()
        ratings_unpacked = []
        for c in ratings_list:
            for t in c:
                ratings_unpacked.append(t)
        #Metrics
        mae = metrics.mean_absolute_error(y_test, ratings_unpacked)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, ratings_unpacked))
        error_y_mean = (mae/np.mean(y_test))*100
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">MAE: {mae:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">RMSE: {rmse:.2f}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-family:Courier; font-size: 20px;">Error in %: {error_y_mean[0]:.2f}%</p>', unsafe_allow_html=True)
        y_test_list = y_test.values.tolist()
        
        y_test_unpacked = []
        for c in y_test_list:
            for t in c:
                y_test_unpacked.append(t)
        
        #Display Results Plot with Plotly
        
        fig3 = make_subplots(specs=[[{"secondary_y": True}]])

        # Add traces
        fig3.add_trace(
            go.Scatter(x=y_test_unpacked, y = ratings_unpacked, name="Predictions",mode = "markers",
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
        title_text=f" Predictions vs True Values"
            )
        st.plotly_chart(fig3)
        #Add model_3 to app
        #Describe project in explore page!!!!