
from scipy.optimize import linprog, minimize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from math import *
from sympy import *
from itertools import *
# выборка будет состоять из доходностей акщий эпл за 5 лет средние за пол года
import pandas as pd
import pandas_datareader.data as web
#import pandas.io.data as web  # Package and modules for importing data; this code may change depending on pandas version
import datetime
import pylab as py
import matplotlib.pyplot as plt
from  stocs import create_series as cs

for i in range(0,12):
    start = datetime.datetime(2016, 1, 1)
    end = datetime.date.today()
    apple = web.DataReader("MSFT", 'google', start, end)
    print(apple)


class FDHGM():
    def __init__(self, Y, X, Y_test, X_test):
        self.Y = Y
        #self.M = self.Y.size
        #self.k = self.Y.size
        self.x = X
        self.X_test = X_test
        self.Y_test = Y_test

    def solve_part_model(self, xi, xj):
        bounds = list([0]*12)
        C_lam = [[0]*5]*2
        XiXj = xi*xj
        C = np.array([len(self.Y),sum(xi),sum(xj), np.dot(xi,xj), np.dot(xi,xi), np.dot(xj,xj), 0, 0, 0, 0, 0, 0])
        A = np.matrix(np.array([[1]*len(self.Y)+[1]*len(self.Y),
             np.ravel([xi, xi]),
             np.ravel([xj, xj]),
             np.ravel([(XiXj), (XiXj)]) ,
             np.ravel([xi ** 2,xi ** 2]),
             np.ravel([xj ** 2,xj ** 2]),
             np.ravel([[-1]*len(self.Y),[1]*len(self.Y)]),
             np.ravel([-np.absolute(xi),np.absolute(xi)]),
             np.ravel([-np.absolute(xj),np.absolute(xj)]),
             np.ravel([-np.absolute(XiXj),np.absolute(XiXj)]),
             np.ravel([-(xi ** 2),(xi ** 2)]),
             np.ravel([-(xj ** 2),xj ** 2])]))
        Y1 = [-1*i for i in self.Y]
        b = np.array(self.Y + Y1)
        for i in range (0, 6):
            bounds[i] = (None, None)
        for i in range (6,12):
            bounds[i] = (0, None)
        #print(tuple(bounds))
        #print(pd.DataFrame(xi.values*xj.values))
        print('c = ')
        print(C)
        print('A = ')
        print(A)
        print('b = ')
        print(b)
        print('bounds = ')
        print(tuple(bounds))
        res = linprog(C, A, b, bounds = bounds,  options={"disp": True})
        print(res)
        C_Lam = res.reshape(6,2)
        с = C_Lam[ :0]
        lam = C_lam[ :1]
        xi = symbols('xi')
        xj = symbols('xj')
        z = [1, xi, xj, xj*xi, xi ** 2, xj ** 2 ]
        y_model = [np.dot(lam,z), np.dot(с ,np.absolute(z))]
        return (y_model)

    def gen_all_combinations(self):
        a = list(combinations(self.x, 2))
        return(a)

    # def find_F_part_models(self, comb, comb_test):
    #     надо сделать все возможные комбинации столбцев и в итоге получить масив из матриц 2 на к и такие же комбинации для тестовой выборки
        #res = list(combinations(x,2))
        # Y_aprox_learn = np.array([])
        # Y_aprox_test = np.array([])
        # Y_model = np.array([])
        # eps = np.array([])
        # n = np.array([])
        #
        # for i in range(1, len(comb)):
        #     criteria = [[]]
        #     Y_model[i] = self.solve_part_model(comb[0, i], comb[1, i])
        #     Y_aprox_learn[i] = Y_model[i].subs(['xi' = comb[0,i], 'xj' = comb[1,i]])
        #     Y_aprox_test[i] = Y_model[i].subs(['xi' = comb_test[0,i], 'xj' = comb_test[1,i]])
        #     eps[i] =  sum((pow(Y_aprox[i] - self.Y_test),2))
        #     n[i] = (1/(self.Y.size+self.Y_test.size))*sum((pow(Y_aprox_learn[i] - self.Y_aprox_test),2))
        #     criteria[i] = [eps[i]+ n[i],i]
        #
        # return # ф лучших Y_model[i], критерий несмещенности

    # def stop(self):
    #
    #     while self.find_F_part_models()[1] < N_k-1:
            # сгенерить комбинации(х)
            # найти f моделей
            # подставить значения моделей = х
data = cs('AAPL')
fdhgm = FDHGM(data[0],data[1],data[2],data[3])
a = fdhgm.gen_all_combinations()
xi = (a[5][0].values)
xj = (a[5][1].values)
print(xi,xj)
# print(a[5][0])
fdhgm.solve_part_model(xi,xj)



















