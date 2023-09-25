# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 06:56:23 2023

@author: Matheus
"""

#Importando bibliotecas
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


#Importação dos dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

classificador = Sequential()
#Criando a camada oculta
#Primeira camada oculta
classificador.add(Dense(units= 8, activation= 'relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30))
classificador.add(Dropout(0.2))
#Segunda camada oculta
classificador.add(Dense(units= 8, activation= 'relu', 
                        kernel_initializer= 'random_uniform'))
classificador.add(Dropout(0.2))
#Criando a camada de saída
classificador.add(Dense(units= 1, activation= 'sigmoid'))

#Compilando a rede com os parâmetros selecionados
classificador.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

#Executando o treinamento da rede
classificador.fit(previsores, classe, batch_size= 10, epochs= 100)

#Fazendo a previsão de novos dados
novo = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178, 
                 0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                 0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                 0.84, 158, 0.363]])
#Executando a previsão
previsao = classificador.predict(novo)
previsao = (previsao > 0.5)