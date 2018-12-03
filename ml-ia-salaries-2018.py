import numpy as np
from sklearn.metrics import accuracy_score #check accuracy
from sklearn.model_selection import train_test_split #randomness
import pandas as pd
import csv 

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self._weights = None
        self._input_size = 0
        self._output_size = 0
        self._b = None
        self._num_samples = 0
        self._lr = learning_rate
        
    def fit(self, X, y, num_epochs = 1, batch_size=None):
        # Check if its a vector.
        if len(y.shape) == 1: 
            y_velho = y.copy()
            num_class = len(set(y_velho))
            print(num_class)
            y = np.zeros((y_velho.shape[0], 2)) #transform it to canonical base
            for i in range(y_velho.shape[0]):
              posi = y_velho[i]
              y[i][posi] = 1        
        # batch size
        if batch_size is None:
            batch_size = X.shape[0]
            
        self._num_samples = X.shape[0]
        self._output_size = y.shape[1]
        self._input_size = X.shape[1]

        #Pesos e neuronios.
        self._b =  np.ones(self._output_size)
        self._weights =  np.random.rand(self._output_size, self._input_size)
        


        #-------------------------------------------------
        # ----------------------TRAINING ALG-----------------
        #-------------------------------------------------



        for i in range(num_epochs):
            E = np.random.randint(0, self._num_samples, batch_size)

            for elemento in E:
                x = X[elemento]
                y_esperado = y[elemento]
                Z = np.dot(self._weights, x) + self._b
                y_chapeu = self._func_ativacao(Z)

                # calculating gradient         
                erro = (y_chapeu  - y_esperado)
                derivada_ativacao = self._deriva_func_ativacao(Z)
                delta =  erro * derivada_ativacao
                                
                # calculating gradient
                for g in range(self._output_size):        
                    gradient_w = (1.0/batch_size) * np.dot(delta[g] , x)
                    gradient_b = (1.0/batch_size) * delta[g]
                    self._weights[g] = self._weights[g] - (self._lr * gradient_w)
                    self._b[g] = self._b[g] - (self._lr * gradient_b)    




    def predict(self, X):
        Z = np.array([ np.dot(self._weights, x) + self._b  for x in X])
        output = self._func_ativacao(Z)
        return np.argmax(output, axis=1)
        
    def _func_ativacao(self, Z):
        return (1.0)/(1.0 + np.e**(-Z))
        
    def _deriva_func_ativacao(self, Z):
        res_ativacao = self._func_ativacao(Z)
        return res_ativacao * (1- res_ativacao)


#-------------------------------------------------
# ----------------------TESTING -----------------
#-------------------------------------------------




# 1. Loading data...
data = pd.read_csv("./salaries1.csv", header=None, sep=';') 
df = pd.DataFrame(data)
df.columns = ['sx','rk','yr','dg', 'yd', 'sl']


#2. setting X and target goal
mapping = {'   male ': 0, ' female ': 1} #fazendo as classes serem um vetor
mapping_02 = {'     full ': 0, 'assistant ': 1, 'associate ': 2}
mapping_03 = {'  masters ': 0, 'doctorate ': 1}
df = df.replace({'sx': mapping, 'rk': mapping_02, 'dg': mapping_03}) 
X = df[['rk','yr','dg','yd', 'sl']]
X = np.asarray(X)
y = df['sx'].values
y = np.asarray(y)
print(y)


#3. Adding randomness to data (Setting 0.2 of data for testing)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.20, random_state=9)


#4. Effectivelly training using perceptron.
clf = Perceptron()
clf.fit(X_treino, y_treino, num_epochs=1000, batch_size=32)


#5. Using the model...
y_chapeu = clf.predict(X_teste)
acc = accuracy_score(y_teste, y_chapeu)

print("acuracia = ", acc)
for i in range(len(y_chapeu)):
    print ("y_experado: %d -- y_obtido: %d " %(y_teste[i], y_chapeu[i]))



