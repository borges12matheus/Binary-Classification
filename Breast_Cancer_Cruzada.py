# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 15:34:20 2023

@author: Matheus
"""

#Importando bibliotecas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

#Importação dos dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#Função que cria a rede
def criarRede():
    classificador = Sequential()
    #Criando a camada oculta
    #Primeira camada oculta
    classificador.add(Dense(units= 16, activation= 'relu', 
                            kernel_initializer= 'random_uniform', input_dim = 30))
    classificador.add(Dropout(0.2))
    #Segunda camada oculta
    classificador.add(Dense(units= 16, activation= 'relu', 
                            kernel_initializer= 'random_uniform'))
    classificador.add(Dropout(0.2))
    #Criando a camada de saída
    classificador.add(Dense(units= 1, activation= 'sigmoid'))
    
    #Criando o otimizador
    otimizador = keras.optimizers.Adam(learning_rate= 0.001,weight_decay= 0.0001, clipvalue= 0.5)
    
    #Executando o treinamento da rede
    classificador.compile(optimizer= otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
    return classificador

#Classificando a rede
classificador = KerasClassifier(build_fn= criarRede,
                                epochs= 100,
                                batch_size= 10)
#Fazendo a validação cruzada
resultados = cross_val_score(estimator = classificador,
                             X = previsores, y = classe,
                             cv = 10, scoring = 'accuracy')
#Calculando a média do percentual de acertos
media = resultados.mean()
#Verificando o desvio padrão
desvio = resultados.std()