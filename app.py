import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')

filename = 'knn_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
    
df = pd.read_excel("rfm.xlsx")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(
    '<style>body{background-color: Blue;}</style>', unsafe_allow_html=True)
st.title("Prediction")

with st.form("my_form"):
    Recency = st.number_input(label='Recency', step=1)
    Frequency = st.number_input(label='Frequency', step=1)
    Monetary = st.number_input(label='Monetary', step=100)

    data = [[Recency, Frequency, Monetary]]
    data = scaler.transform(data)

    submitted = st.form_submit_button("Submit")

if submitted:
    prediction = loaded_model.predict(data)
    prediction = prediction[0]
    print('Data berada di Cluster',prediction)

    cluster_df1 = df[df['Segment'] == prediction]
    for c in cluster_df1.drop(['Segment'], axis=1):
        grid = sns.FacetGrid(cluster_df1, col='Segment')
        grid = grid.map(plt.hist, c)
    plt.show()