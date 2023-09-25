#Importando bibliotecas
import pandas as pd
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

#Salvando a rede neural
classificador_json = classificador.to_json()
with open('classificado_breast.json', 'w') as json_file:
    json_file.write(classificador_json)
#Salvando os pesos da rede
classificador.save_weights('classificador_breast.h5')    