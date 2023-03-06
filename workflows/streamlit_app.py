"""
Created on Sat March 05 14:55:49 2023

@author: Vladimir Burlay
"""
import streamlit as st
import pandas as pd
import scripts_caravan_class
import plotly.express as px
import urllib

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/vburlay/anw_feld_ba/main/workflows/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")



st.sidebar.title("Control Panel")

with st.sidebar:
    add_selectbox = st.selectbox("App-Mode", ["Application start","Show the source code"])
    add_radio = st.radio( "Choose a model",("Logistic Regression", "Keras"))


if add_selectbox  == "Application start" :
    lg_pred, dl_pred = scripts_caravan_class.main()
    data_ml = pd.DataFrame(data = lg_pred)
    data_dl = pd.DataFrame(data = dl_pred)
    st.title("Caravan insurance")

    tab1, tab2, tab3 = st.tabs(["Countplot of the results", "Result Tabular","Individual results"])
    with tab1:
        if add_radio == "Logistic Regression":
            st.bar_chart(data=data_ml.loc[:,['Yes_Prob','predicted']], x = 'predicted', width=1000,height=500)
        elif add_radio == "Keras":
            st.bar_chart(data=data_dl.loc[:, ['Yes_Prob', 'predicted']], x='predicted', width=1000, height=500)
    with tab2:
        if add_radio == "Logistic Regression":
            fig = px.scatter(data_ml.loc[:,['Yes_Prob','predicted']], width=1000, height=650)
            st.plotly_chart(fig)
        elif add_radio == "Keras":
            fig = px.scatter(data_dl.loc[:,['Yes_Prob','predicted']], width=1000, height=650)
            st.plotly_chart(fig)
        with tab3:
            if add_radio == "Logistic Regression":
                st.dataframe(data_ml.drop(columns=['ID','Yes']),width=1200,height=600)
            elif add_radio == "Keras":
                st.dataframe(data_dl.drop(columns=['ID','Yes']), width=1200, height=600)


elif add_selectbox == "Show the source code":
    readme_text = st.markdown(get_file_content_as_string("streamlit.md"))


