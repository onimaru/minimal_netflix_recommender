import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import pickle
import numpy as np

st.set_page_config(layout="wide")

df = pd.read_csv("data/normalized_netflix.csv")
model = pickle.load( open( "data/model.pickle", "rb" ) )
X = np.load("data/features_space.npy")

def find_item_by_title_token(name,df,n=10):
    return df[df["title"].str.contains(name)]["title"].iloc[:n].to_dict()

def get_recommendations_from_inputs(input_ids,df):
    recommendations_dict = {}
    for idx in input_ids:
        recommended_ids = model.kneighbors(X[idx].reshape(1,-1),return_distance=False)[0].tolist()
        x = df.iloc[recommended_ids]["title"].tolist()
        recommendations_dict[x[0]] = x[1:]
    return recommendations_dict

def display_recommendations(input_ids,df):
    recommendations_dict = get_recommendations_from_inputs(input_ids,df)
    msg = '| input | rec 01 | rec 02 | rec 03 | rec 04 | rec 05 | rec 06 | rec 07 | rec 08 | rec 09 |\n'
    msg += '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for input_x, recs in recommendations_dict.items():
        msg += f"| **{input_x}** |{'|'.join(recs)}|\n"

    st.markdown(msg)

def show_recommendations(title_token,df=df):
    retrieved_items = find_item_by_title_token(title_token,df,n=20)
    display_recommendations(list(retrieved_items.keys()),df)
    
    
st.title("Minimal Netflix Recommender System")
st.write("Os dados usados são do Netflix americano de setembro de 2021. Então a recomendação pode estar desatualizada.")
title_token = st.text_input('Palavra chave', "dark")
show_recommendations(title_token.lower())
