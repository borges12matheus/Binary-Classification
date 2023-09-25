# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 10:25:22 2023

@author: Matheus
"""
#Importação dos dados
import pandas as pd
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#Fazendo a divisão da base de dados
from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

#Criando a rede
import keras
from keras.models import Sequential
from keras.layers import Dense
classificador = Sequential()
#Criando a camada oculta
#Primeira camada oculta
classificador.add(Dense(units= 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform', input_dim = 30))
#Segunda camada oculta
classificador.add(Dense(units= 16, activation= 'relu', 
                        kernel_initializer= 'random_uniform'))

#Criando a camada de saída
classificador.add(Dense(units= 1, activation= 'sigmoid'))

#Criando o otimizador
otimizador = keras.optimizers.Adam(learning_rate= 0.001,weight_decay= 0.0001, clipvalue= 0.5)

#Executando o treinamento da rede
classificador.compile(otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size=10, epochs= 100)

#Analizando os valores dos pesos
pesos0 = classificador.layers[0].get_weights()
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()

#Testando a rede
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes>0.5)

#Fazendo a avaliação da resposta da rede
from sklearn.metrics import confusion_matrix, accuracy_score
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

#Usando o keras
resultado = classificador.evaluate(previsores_teste, classe_teste)