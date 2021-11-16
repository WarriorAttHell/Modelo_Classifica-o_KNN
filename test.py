
import joblib as jb
import numpy as np
import streamlit as st
import os
from joblib import dump, load

st.header('Modelo de Classificação')
st.subheader('Construido em Python')
st.markdown('Insira as informações para efetuar as previsões')


alturaSepala = st.slider('Informe a altura da sépata em cm', 0.0, 10.0)
st.write(alturaSepala)

larguraSepala = st.slider('Informe a largura da sépata em cm', 0.0, 5.0)
st.write(larguraSepala)

alturaPetala = st.slider('Informe a altura da petala em cm', 0.0, 7.0)
st.write(alturaPetala)

larguraPetala = st.slider('Informe a largura da petala em cm', 0.0, 3.0)
st.write(larguraPetala)

patch = 'C:/Users/deand/OneDrive/estudy/Iris/modelo_KNN.pk1'

if (os.path.exists('modelo_KNN.pk1')):
    modelo = load('modelo_KNN.pk1')
    botao = st.button('Efetuar previsão')
    if(botao):
        listaValores = np.array([[alturaSepala,larguraSepala,alturaPetala,larguraPetala]])
        resultado = modelo.predict(listaValores)
        if(resultado[0] == 1):
            st.write('Iris-Setosa')
        elif (resultado[0] == 2):
            st.write('Iris-Versicolour')
        else:
            st.write('Iris-Virginica')
else:
    st.error('Erro ao carregar o modelo preditivo. Contate o administrator.')