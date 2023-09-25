# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 08:00:08 2023

@author: Matheus
"""
import numpy as np
from sklearn import datasets

#Definição da função de ativação sigmoid
def sigmoid(soma):
    return 1/(1 + np.exp(-soma))

def sigmoidDerivada(sig):
    return sig*(1-sig)

#Carregando base de dados breast cancer
base = datasets.load_breast_cancer()

#Matrizes de entrada e saída
entradas = base.data

valorSaidas = base.target
saidas = np.empty([569, 1],dtype = int)
for i in range(569):
    saidas[i] =  valorSaidas[i]

#Matriz de pesos entre a camada de entrada e oculta
pesos0 = 2*np.random.random((30,5)) - 1
pesos1 = 2*np.random.random((5,1)) - 1

epocas = 1000
taxaAprendizagem = 0.3
momento = 1

for j in range(epocas):
    camadaEntrada = entradas
    #Aplicação da função soma e função de ativação camada entrada - oculta
    somaSinapse0 = np.dot(camadaEntrada, pesos0)
    camadaOculta = sigmoid(somaSinapse0)
    #Aplicação da função soma e função de ativação camada oculta - saída
    somaSinapse1 = np.dot(camadaOculta, pesos1)
    camadaSaida = sigmoid(somaSinapse1)
    
    #Calculando o erro da saída e erro médio
    erroSaida = saidas - camadaSaida
    erroMedio = np.mean(np.abs(erroSaida))
    print("Erro: "+str(erroMedio))
    
    #Calculando delta de saída
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroSaida*derivadaSaida
    
    #Calculando delta da camada oculta
    produtoPesoDelta = deltaSaida.dot(pesos1.T)
    derivadaOculta = sigmoidDerivada(camadaOculta)
    deltaOculta = derivadaOculta*produtoPesoDelta
    
    #Atualização de pesos camada oculta
    pesosNovo1 = (camadaOculta.T).dot(deltaSaida)
    pesos1 = (pesos1*momento) + (pesosNovo1*taxaAprendizagem)
    
    #Atualização de pesos camada entrada
    pesosNovo0 = (camadaEntrada.T).dot(deltaOculta)
    pesos0 = (pesos0*momento) + (pesosNovo0*taxaAprendizagem)
    