"""
Created on Sat March 05 14:55:49 2023

@author: Vladimir Burlay
"""
import streamlit as st
import pandas as pd
import scripts_caravan_class
import plotly.express as px
from streamlit_imagegrid import streamlit_imagegrid
import requests
import urllib

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/vburlay/ser_str/master/workflow/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")



st.sidebar.title("Control Panel")

with st.sidebar:
    add_selectbox = st.selectbox("App-Mode", ["Application start","Show the source code"])
    add_radio = st.radio(
        "Choose a model",
        ("Logistic Regression", "Keras")   )


if add_selectbox  == "Application start" :
    lg_pred, dl_pred = scripts_caravan_class.main()
    data_ml = pd.DataFrame(data = lg_pred)
    data_dl = pd.DataFrame(data = dl_pred)
    st.title("Caravan insurance")

    tab1, tab2, tab3 = st.tabs(["Countplot of the results", "Result Tabular","Individual results"])
    with tab1:
        fig = px.bar(data_ml['Yes_Prob'],width=1000,height=500)
        st.plotly_chart(fig)
    with tab2:
        fig = px.scatter(data_ml.drop(columns=['ID']), width=1000, height=650)
        st.plotly_chart(fig)
        with tab3:
            st.dataframe(data_ml,width=1200,height=600)


elif add_selectbox == "Show the source code":
    readme_text = st.markdown(get_file_content_as_string("streamlit.md"))


