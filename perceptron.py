import random
import os
import csv
import pandas as pd
import sklearn.model_selection as ms
import sklearn.metrics as mtr
import matplotlib.pyplot as plt

# Equipe :
# Mateus Martins Pereira - 17.1.8109
# Gabriel Batista - 17.1.8083

class Perceptron:

    # Inicializacao do objeto Perceptron
    def __init__(self,learn_rate=0.01, epoch_number=1000, bias=-1):
        self.sample = []
        self.exit = []
        self.datasetTreinamento = []
        self.exitDatasetTreinamento = []
        self.datasetTeste = []
        self.exitDatasetTeste = []
        self.learn_rate = learn_rate
        self.epoch_number = epoch_number
        self.bias = bias
        self.weight = []
        # fill() preenche os atributos sample e exit
        self.fill()
        # alocaDataset() distribui de modo aleatório 60% do dataset para treinamento e 40% para teste, utilizando a lib sklearn
        self.alocaDataset()
        # define número de de registros e colunas para cada dataset (número de colunas é sempre igual)
        self.number_sample = len(self.sample)
        self.number_treinamento = len(self.datasetTreinamento)
        self.number_teste = len(self.datasetTeste)
        self.col = len(self.sample[0])

    # Funcao de Treinamento do Perceptron (Metodo Gradiente Descendente)
    def trannig(self):
        for element in self.datasetTreinamento:
            element.insert(0, self.bias)
        
        # Inicializa os pesos w aleatoriamente
        for i in range(self.col):
           self.weight.append(random.random())

        # Insere peso da entrada de polarizacao(bias)
        self.weight.insert(0, self.bias)

        epoch_count = 0
        #Metodo do Gradiente Descendente para ajuste dos pesos do Perceptron
        while True:
            erro = False
            for i in range(self.number_treinamento):
                u = 0
                for j in range(self.col + 1):
                    u = u + self.weight[j] * self.datasetTreinamento[i][j]
                y = self.sign(u)
                if y != self.exitDatasetTreinamento[i]:
                    for j in range(self.col + 1):
                        self.weight[j] = self.weight[j] + self.learn_rate * (self.exitDatasetTreinamento[i] - y) * self.datasetTreinamento[i][j]
                    erro = True
            
            print('Epoca: \n',epoch_count)
            epoch_count = epoch_count + 1
            
            # Verifica se atinge o limite de épocas (critério de parada) ou se não há mais erro 
            if(epoch_count == 500 or erro == False):
                break

    # Funcao que testa, por meio de uma matrix de confusão, o quão bom é o classificador
    def test(self):
        predicted = []
        for element in self.datasetTeste:
            element.insert(0, self.bias)
            u = 0
            for i in range(self.col + 1):
                u = u + self.weight[i] * element[i]
            y = self.sign(u)
            predicted.append(y)
        matriz = mtr.confusion_matrix(self.exitDatasetTeste, predicted)
        specificity = matriz[0][0]/(matriz[0][0] + matriz[0][1])
        sensitivity = matriz[1][1]/(matriz[1][0] + matriz[1][1]) 
        print("=== Matriz de confusao ===")
        print(matriz)
        print("=== Especificidade ===")
        print(specificity)
        print("=== Sensibilidade ===")
        print(sensitivity)

# Funcao de Ativacao
    def sign(self, u):
        return 1 if u >= 0 else 0

# Funcao que preenche sample e exit
    def fill(self):
        base_path = os.path.abspath(os.path.dirname(__file__))
        path = os.path.join(base_path, "dataset_spam.dat")
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=',')
            for linha in csv_reader:
                lista = []
                count = 1
                for valor in linha:
                    # Verifica se é o último elemento da lista, para adicionar à lista de saida, identificando se é spam ou não
                    if (count == len(linha)):
                        self.exit.append(float(valor))
                    else:
                        lista.append(float(valor))
                    count += 1
                self.sample.append(lista)

# Funcao que preenche os datasets de treinamento e teste e seus respectivos exits
    def alocaDataset(self):
        self.datasetTreinamento,self.datasetTeste,self.exitDatasetTreinamento, self.exitDatasetTeste = ms.train_test_split(self.sample, self.exit, train_size=0.6)

# Inicializa o Perceptron
network = Perceptron(learn_rate=0.1, epoch_number=1000, bias=-1)
# Chamada ao treinamento
network.trannig()
# Chama ao teste
network.test()

