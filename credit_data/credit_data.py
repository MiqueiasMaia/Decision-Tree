# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Decision-Tree/credit_data/credit_dt.csv')

previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 1:4])
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier, export
classificador = DecisionTreeClassifier(criterion='entropy', random_state=0)
classificador.fit(previsores_treinamento, classe_treinamento)

previsoes = classificador.predict(previsores_teste)

from sklearn.metrics import confusion_matrix, accuracy_score

precisao = accuracy_score(classe_teste, previsoes)
print('precisao: ')
print(precisao)
matriz = confusion_matrix(classe_teste, previsoes)
print(matriz)
export.export_graphviz(classificador,out_file='arvore2.dot',feature_names=['Income','Age','Loan'],class_names=['0','1'], filled=True,leaves_parallel=True)
