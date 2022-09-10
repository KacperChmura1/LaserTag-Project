import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64

def proc(acc):
    acc = acc[:-1]
    return acc
@st.cache
def load_data():
    df = pd.read_csv("Stats.csv",index_col = 0)
    df["Accuracy"] = df["Accuracy"].apply(proc).apply(float)

    return df

df = load_data()

def show_explore_page():
    st.title("About Project")
    paragraph1 = "This project is based on machine learning models that are used for prediction. In this particular case of predicting the rating of players in the LaserTag games, it is a bit of an art for art's sake, because we know the exact formula for the real Rating. However, the project aims to prove how modern artificial intelligence copes with prediction."
    paragraph2 = "As we can see, the above pattern uses only three features(Hits, DMG_Get, Accuracy), while our model uses five(Hits, DMG_Get, Accuracy, Deaths, Shot fired). These six features are the basic statistics, and they were used to confuse our model. In the following, also a model using only these three features (cheated model) will be added and both will be compared."
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph1} </p>', unsafe_allow_html=True)
    #Equation
    st.markdown(f'<p style="font-family:Courier; font-size: 30px;"> Real Formula: </p>', unsafe_allow_html=True)

    st.latex(r'''Rating = \frac{Hits}{DMG get + 1}  * \frac{Accuracy + 100}{3}''')
    st.markdown(f"<p>\n</p>", unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 30px;"> Model and Cheating Model </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-family:Courier; font-size: 20px;"> {paragraph2} </p>', unsafe_allow_html=True)

    
    
    #\frac{Hits}{DMG get + 1 } * \frac{Accuracy + 100}{3}
    
    #fig1, ax1 = plt.subplots()
    #ax1.plot(df)
    #ax1.axis("equal")

    #fig = plt.figure()
    #plot = sns.lineplot(x = df["Rating"], y = df["Accuracy"])
    #st.pyplot(fig)