"""
Created on Sat March 05 14:55:49 2023

@author: Vladimir Burlay
"""
import streamlit as st
import plotly.express as px
import urllib

import os.path
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import keras

def probability(probs, y_test):
    y_pred_df = pd.DataFrame(probs)
    y_pred_1 = y_pred_df.iloc[:, [1]]

    y_test_df = pd.DataFrame(y_test)

    # Put the index as ID column, remove index from both dataframes and combine them
    y_test_df["ID"] = y_test_df.index
    y_pred_1.reset_index(drop=True, inplace=True)
    y_test_df.reset_index(drop=True, inplace=True)
    y_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)
    y_pred_final = y_pred_final.rename(columns={1: "Yes_Prob", "CARAVAN": "Yes"})
    y_pred_final = y_pred_final.reindex(["ID", "Yes", "Yes_Prob"], axis=1)

    numbers = [float(x) / 10 for x in range(10)]
    y_pred_final['predicted'] = y_pred_final.Yes_Prob.map(lambda x: 1 if x > 0.1 else 0)
    for i in numbers:
        y_pred_final[i] = y_pred_final.Yes_Prob.map(lambda x: 1 if x > i else 0)
    return y_pred_final


def ml(X_test, y_test):
    mm_scale = preprocessing.MinMaxScaler()
    X_test[X_test.columns] = mm_scale.fit_transform(X_test[X_test.columns])

 #   path_dir = str(Path(__file__).resolve().parent.parent)
 #   path_model = os.path.join(path_dir, 'tests\models\ml_model.sav')
    path_model = 'models/ml_model.sav'
    if os.path.isfile(path_model):  #
        reg = joblib.load(path_model)

    pca = PCA(n_components=0.95)
    X_test_reduced = pca.fit_transform(X_test)
    X_test_reduced = X_test_reduced[:, 0:34]

    probs = reg.predict_proba(X_test_reduced)
    y_ml_pred = probability(probs, y_test)
    return y_ml_pred


def dl(X_test, y_test):
#    path_dir = str(Path(__file__).resolve().parent.parent)
#    path_model = os.path.join(path_dir, 'tests\models\cnn_model.h5')
    path_model ='models/cnn_model.h5'

    if os.path.isfile(path_model):  #
        x_testcnn = np.expand_dims(X_test, axis=(2))

    model_cnn = keras.models.load_model(path_model)
    probs = model_cnn.predict(x_testcnn)

    y_dl_pred = probability(probs, y_test)
    return y_dl_pred


def main():
#    path_dir = str(Path(__file__).resolve().parent.parent)
#    path_to_file = path_dir + '\date\caravan-insurance-challenge.csv'
    path_to_file = 'https://raw.githubusercontent.com/vburlay/anw_feld_ba/main/date/caravan-insurance-challenge.csv'
    data = pd.read_csv(path_to_file)

    test = data[data['ORIGIN'] == 'test']
    y_test = test['CARAVAN']
    X_test = test.drop(['ORIGIN', 'CARAVAN'], axis=1)
    lg_pred = ml(X_test, y_test)
    dl_pred = dl(X_test, y_test)
    return lg_pred, dl_pred

def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/vburlay/anw_feld_ba/main/workflows/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")



st.sidebar.title("Control Panel")

with st.sidebar:
    add_selectbox = st.selectbox("App-Mode", ["Application start","Show the source code"])
    add_radio = st.radio( "Choose a model",("Logistic Regression", "Keras"))


if add_selectbox  == "Application start" :
    lg_pred, dl_pred = main()
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


