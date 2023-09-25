#Importando bibliotecas
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


#Importação dos dados
previsores = pd.read_csv('entradas_breast.csv')
classe = pd.read_csv('saidas_breast.csv')

#Função que cria a rede
def criarRede(optimizer, loss, kernel_initializer, activation, neurons):
    classificador = Sequential()
    #Criando a camada oculta
    #Primeira camada oculta
    classificador.add(Dense(units= neurons, activation= activation, 
                            kernel_initializer= kernel_initializer, input_dim = 30))
    classificador.add(Dropout(0.2))
    #Segunda camada oculta
    classificador.add(Dense(units= neurons, activation= activation, 
                            kernel_initializer= kernel_initializer))
    classificador.add(Dropout(0.2))
    #Criando a camada de saída
    classificador.add(Dense(units= 1, activation= 'sigmoid'))
    
    #Executando o treinamento da rede
    classificador.compile(optimizer= optimizer, loss = loss, metrics = ['binary_accuracy'])
    return classificador

#Criando o classificador
classificador = KerasClassifier(model= criarRede)
parametros = {'batch_size': [10, 30],
              'epochs': [50, 100],
              'model__optimizer': ['adam','sgd'],
              'model__loss': ['binary_crossentropy','hinge'],
              'model__kernel_initializer': ['random_uniform','normal'],
              'model__activation': ['relu', 'tanh'],
              'model__neurons':[16,8]}

#Fazendo a busca pelos melhores parâmetros usando o Grid Search
grid_search = GridSearchCV(estimator = classificador,
                           param_grid= parametros,
                           scoring= 'accuracy',
                           cv = 5)
#Fazendo o treinamento com os parâmetros
grid_search = grid_search.fit(previsores,classe)

#Guardando os melhores parâmetros
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_