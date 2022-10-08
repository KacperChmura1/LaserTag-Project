import streamlit as st
import pandas as pd
import seaborn as sns
import base64
from PIL import Image
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
def load_model(model_name):
    with open (model_name, "rb") as file:
        data = pickle.load(file)
    return data
#Loading ML models
X_test = pd.read_csv("data/X_test.csv",index_col = 0)
X_test_3 = pd.read_csv("data/X_test_3.csv",index_col = 0)
y_test = pd.read_csv("data/y_test.csv",index_col = 0)
X_train_ann  = pd.read_csv("data/X_train_ann.csv",index_col = 0)
X_train_ann_3  = pd.read_csv("data/X_train_ann_3.csv",index_col = 0)
X_test_ann_3 = pd.read_csv("data/X_test_ann_3.csv",index_col = 0)
Linear_reg = load_model("Models/linear_regression.pkl")
Linear_reg_3 = load_model("Models/linear_regression_3.pkl")
forest = load_model("Models/forest.pkl")
forest_3 = load_model("Models/forest_3.pkl")
gradient = load_model("Models/gradient.pkl")
gradient_3 = load_model("Models/gradient_3.pkl")
SVRLinear = load_model("Models/SVRLinear.pkl")
SVRLinear_3 = load_model("Models/SVRLinear_3.pkl")
KNN = load_model("Models/KNN.pkl")
KNN_3 = load_model("Models/KNN_3.pkl")
#Loading DeepLearning models
def load_ann_model(model_name):
    model = tf.keras.models.load_model(model_name)
    return model
ann = load_ann_model("Models/ann.hdf5")
ann_3 = load_ann_model("Models/ann_3.hdf5")
#Delete % from Accuracy column and transform str to float
def proc(acc):
    acc = acc[:-1]
    return acc
def load_data():
    df = pd.read_csv("Stats.csv",index_col = 0)
    df["Accuracy"] = df["Accuracy"].apply(proc).apply(float)
    return df
df = load_data()
#Function to ploting model result
def plots(model1,model2,name,ann_name):
    X_test = pd.read_csv("data/X_test.csv",index_col = 0)
    X_test_3 = pd.read_csv("data/X_test_3.csv",index_col = 0)
    if ann_name == "ann":
        ann = load_ann_model("Models/ann.hdf5")
        ann_3 = load_ann_model("Models/ann_3.hdf5")
        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train_ann)
        X_test = scaler.transform(X_test)
        scaler2 = MinMaxScaler()
        X_train_s_3 = scaler2.fit_transform(X_train_ann_3)
        X_test_3 = scaler2.transform(X_test_3)
        ratings = model1.predict(X_test)
        ratings2 = model2.predict(X_test_3)
        ratings_unpacked = []
        for c in ratings:
            for t in c:
                ratings_unpacked.append(t)
        ratings = ratings_unpacked
        ratings2_unpacked = []
        for c in ratings2:
            for t in c:
                ratings2_unpacked.append(t)
        ratings2 = ratings2_unpacked
    else: 
        ratings = model1.predict(X_test)
        ratings2 = model2.predict(X_test_3)
    y_test_list = y_test.values.tolist()
    if ann_name != "ann":
        ratings_list = ratings.tolist()
        ratings_list2 = ratings2.tolist()
    else:
        ratings_list = ratings
        ratings_list2 = ratings2
    y_test_unpacked = []
    for c in y_test_list:
        for t in c:
            y_test_unpacked.append(t)      
    fig3 = make_subplots(rows=1, cols=2,
                    specs=[[{"secondary_y": True}, {"secondary_y": True}]],
                           subplot_titles=(f"{name}", f"Cheated {name}"))
    # Add traces
    fig3.add_trace(
        go.Scatter(x=y_test_unpacked, y = np.round(ratings_list,2), name="Predictions",mode = "markers",
        marker=dict(
        color='rgba(135, 206, 250, 0.6)',
        size=6
            )
        ),      
        row=1, col=1, secondary_y=False),
    fig3.add_trace(
        go.Scatter(x=y_test_unpacked, y = y_test_unpacked, name="Real Ratings"),
        row=1, col=1, secondary_y=True)
    fig3.add_trace(
        go.Scatter(x=y_test_unpacked, y = np.round(ratings_list2,2), name="Cheated Predictions",mode = "markers",
        marker=dict(
        color='rgba(135, 206, 150, 0.6)',
        size=6
            )
        ),      
        row=1, col=2, secondary_y=False),
    fig3.add_trace(
        go.Scatter(x=y_test_unpacked, y = y_test_unpacked, name="Real Ratings"),
        row=1, col=2, secondary_y=True)
        # Add figure title
    fig3.update_layout(
    width=900,
    height = 500
        )
    st.markdown(f'<p style="font-family:Courier; font-size: 40px;text-align:center;margin-top:50px;"> {name} </p>', unsafe_allow_html=True)
    st.plotly_chart(fig3)
    with st.container():
        col1, col2 = st.columns(2)
        mae = metrics.mean_absolute_error(y_test, ratings)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, ratings))
        error_y_mean = (mae/np.mean(y_test))*100
        mae2 = metrics.mean_absolute_error(y_test, ratings2)
        rmse2 = np.sqrt(metrics.mean_squared_error(y_test, ratings2))
        error_y_mean2 = (mae2/np.mean(y_test))*100
        with col1:
            st.markdown(f'<p style="font-family:Courier; font-size: 30px;text-align:center;"> Regular Model </p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:Courier; font-size: 20px;color:Green;">MAE: {mae:.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:Courier; font-size: 20px;color:Green;">RMSE: {rmse:.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:Courier; font-size: 20px;color:Green;">Error in %: {error_y_mean[0]:.2f}%</p>', unsafe_allow_html=True)          
        with col2:
            st.markdown(f'<p style="font-family:Courier; font-size: 30px;text-align:center;"> Cheated Model </p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:Courier; font-size: 20px;color:Red;">MAE: {mae2:.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:Courier; font-size: 20px;color:Red;">RMSE: {rmse2:.2f}</p>', unsafe_allow_html=True)
            st.markdown(f'<p style="font-family:Courier; font-size: 20px;color:Red;">Error in %: {error_y_mean2[0]:.2f}%</p>', unsafe_allow_html=True)
#Main function to display
def show_explore_page():
    st.title("About Project")
    #Paragraphs
    paragraph1 = "This project is based on machine learning models that are used for prediction. In this particular case of predicting the rating of players in the LaserTag games, it is a bit of an art for art's sake, because we know the exact formula for the real Rating. However, the project aims to prove how modern artificial intelligence copes with prediction."
    paragraph2 = "As we can see, the above pattern uses only three features(Hits, DMG_Get, Accuracy), while our model uses five(Hits, DMG_Get, Accuracy, Deaths, Shot fired). These five features are the basic statistics, and they were used to confuse our model. In the following, also a model using only these three features (cheated model) will be added and both will be compared."
    paragraph3 = "Let's see how the models and their cheated versions can handle the prediction."
    paragraph4 = "Ann did the best, he has almost twice as good results as the second Gradient boost!"
    paragraph5 = """As we can see in most cases, the cheated models lost to the regular models. Why is this happening? """
    paragraph6 = "I will leave this as an open question."
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph1} </p>', unsafe_allow_html=True)
    #Equation
    st.markdown(f'<p style="font-family:Courier; font-size: 30px;text-align:center;"> Real Formula </p>', unsafe_allow_html=True)
    st.latex(r'''Rating = \frac{Hits}{DMG get + 1}  * \frac{Accuracy + 100}{3}''')
    st.markdown(f"<p>\n</p>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 30px;text-align:center;"> Model and Cheating Model </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph2} </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 30px;text-align:center;"> Comparison of models </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph3} </p>', unsafe_allow_html=True)
    #Display plots
    plots(ann,ann_3,"ANN","ann")
    plots(Linear_reg,Linear_reg_3,"Linear Regression",0)
    plots(forest,forest_3,"Random Forest",0)
    plots(gradient,gradient_3,"Gradient Boost",0)
    plots(SVRLinear,SVRLinear_3,"SVR Linear",0)
    plots(KNN,KNN_3,"KNN",0)
    #Display paragraphs
    st.markdown(f'<p style="font-family:Courier; font-size: 30px;text-align:center;"> Thoughts</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;margin-top:50px;"> {paragraph4} </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph5} </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph6} </p>', unsafe_allow_html=True)
