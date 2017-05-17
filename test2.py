from scipy.optimize import linprog, minimize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from math import *
from sympy import *
import time

KO =np.array([43.59,   43.15,   42.44,   41.96,  41.57, 41.46,  40.3,    42.40,   42.32,   43.43,   43.63,   45.33,   44.60])
APL = np.array([ 156.10,    143.65 ,    143.66 ,   136.99,    121.35,     115.82,     110.52,    113.54,     113.05, 106.10, 104.21, 95.60, 99.86])
MSFT = np.array([68.38,    68.46,    65.86,  63.98,    64.65,   62.14,  60.26,   59.92,     57.60,     57.46,   56.68, 51.17 , 53.00])
INTL = np.array([35.53,  36.15 ,  36.07,   36.20,    36.82,  36.27,   34.70,   34.87,   37.75,   35.89,   34.86,  32.80,  31.59])
MDC =  np.array([145.36,139.93,129.61,127.65, 122.57,121.72, 119.27, 112.57, 115.36, 115.66,  117.65,  120.34, 122.06])
for i in range (1,KO.size):
    KO[i-1] = ((KO[i-1]-KO[i])/KO[i])
    INTL[i - 1] = ((INTL[i - 1] - INTL[i]) / INTL[i])
    MDC[i - 1] = ((MDC[i - 1] - MDC[i]) / MDC[i])
    APL[i-1] = ((APL[i - 1] - APL[i]) / APL[i])
    MSFT[i-1] = ((MSFT[i - 1] - MSFT[i]) / MSFT[i])
r_KO = (np.sum(KO[:-1]))/(KO.size-1)
r_APL = (np.sum(APL[:-1]))/(APL.size-1)
r_MSFT = (np.sum(MSFT[:-1]))/(MSFT.size-1)
r_MDC = (np.sum(MDC[:-1]))/(MDC.size-1)
r_INTL = (np.sum(INTL[:-1]))/(INTL.size-1)
r1_KO = r_KO - 0.02
r2_KO = r_KO + 0.02
r1_MDC = r_MDC - 0.02
r2_MDC = r_MDC + 0.02
r1_INTL = r_INTL - 0.02
r2_INTL = r_INTL + 0.02
r1_APL = r_APL - 0.02
r2_APL = r_APL + 0.02
r1_MSFT = r_MSFT - 0.02
r2_MSFT = r_MSFT + 0.02
print (r_KO, r_APL, r_MSFT, r_INTL, r_MDC)
class Portfolio():
    def __init__(self, x, r, r1, r2, r_star, bet):
        self.x = np.array(x)
        self.r = np.array(r)
        self.r1 = np.array(r1)
        self.r2 = np.array(r2)
        self.r_star = r_star
        self.bet = bet

    def calculate_xi(self):

        if self.bet == 0:
            return self.x_betta_eq0()
        elif self.bet == 1:
            return self.x_betta_eq1()
        else:
            return self.x_betta()

    def x_betta_eq0(self):
        print(linprog(self.r, [(-1) * self.r1], (-1) * self.r_star, [[1] * self.r.size], 1))

    def x_betta_eq1(self):
        print(linprog(self.r, [self.r1], self.r_star, [[1] * self.x.size], 1))

    def betta(self):
        self.x = symbols('x[0:%d]' % self.r.size)
        A = log((np.dot(self.x, self.r) - self.r_star) / (np.dot(self.x, self.r) - np.dot(self.x, self.r1)))
        B = ((self.r_star - np.dot(self.x, self.r1) + A*(np.dot(self.x, self.r) - self.r_star)) / (np.dot(self.x, self.r2) - np.dot(self.x, self.r1)))
        return (B - self.bet)

    def betta2(self):
        self.x = symbols('x[0:%d]' % self.r.size)
        A = log((self.r_star-np.dot(self.x, self.r)) / (np.dot(self.x, self.r2) - np.dot(self.x, self.r1)))
        B = ((self.r_star - np.dot(self.x, self.r1) - A * (self.r_star-np.dot(self.x, self.r))) / (
        np.dot(self.x, self.r2) - np.dot(self.x, self.r1)))
        return (B - self.bet)

    def sub_betta(self, x):
        self.x = symbols('x[0:%d]' % self.r.size)
        replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
        return self.betta().subs(replacements)

    def sub_betta2(self,x):
        self.x = symbols('x[0:%d]' % self.r.size)
        replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
        return self.betta2().subs(replacements)

    def find_betta(self, x):
        A = log((np.dot(x, self.r) - self.r_star) / (np.dot(x, self.r) - np.dot(x, self.r1)))
        B = ((self.r_star - np.dot(x, self.r1) + A * (np.dot(x, self.r) - self.r_star)) / (
        np.dot(x, self.r2) - np.dot(x, self.r1)))
        return B

    def find_betta2(self, x):
        A = log((self.r_star - np.dot(x, self.r)) / (np.dot(x, self.r2) - np.dot(x, self.r1)))
        B = ((self.r_star - np.dot(x, self.r1) - A * (self.r_star - np.dot(x, self.r))) / (
            np.dot(x, self.r2) - np.dot(x, self.r1)))
        return B


    def func_x0(self,x, r1,r2 ,r11, r12, r21, r22, b):
           f = (1/((x*r21+(1-x)*r22 - (x*r11+(1-x)*r12))))*((self.r_star-(x*r11+(1-x)*r12)+(x*r1+(1-x)*r2-self.r_star)*log((x*r1+(1-x)*r2-self.r_star)/(x*r1+(1-x)*r2 - (x*r11+(1-x)*r12))))) - b
           return f

    def func_x0_2(self, x, r1,r2 ,r11, r12, r21, r22, b):
           f = (1/((x*r21+(1-x)*r22 - (x*r11+(1-x)*r12))))*((self.r_star-(x*r11+(1-x)*r12)-(self.r_star-(x*r1+(1-x)*r2))*log((x*r1+(1-x)*r2-self.r_star)/(x*r21+(1-x)*r22 - (x*r11+(1-x)*r12))))) - b
           return f

    def diff_betta(self,x):
        dbetta = [0]*self.r.size
        dbetta1 = [0] * self.r.size
        for i in range(0,self.r.size):
            dbetta[i] = (diff(self.betta(), self.x[i]))
            replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
            dbetta1[i] = dbetta[i].subs(replacements)
        return dbetta1

    def diff_betta2(self, x):
        dbetta = [0] * self.r.size
        dbetta1 = [0] * self.r.size
        for i in range(0, self.r.size):
            dbetta[i] = (diff(self.betta2(), self.x[i]))
            replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
            dbetta1[i] = dbetta[i].subs(replacements)
        return dbetta1

    def f(self):
        self.x = symbols('x[0:%d]' % self.r.size)
        f = -1*(np.dot(self.x, self.r))
        return f

    def sub_f(self, x):
        self.x = symbols('x[0:%d]' % self.r.size)
        replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
        return self.f().subs(replacements)



    def diff_f(self,x):
        self.x = symbols('x[0:%d]' % self.r.size)
        df = [0] * self.r.size
        dff = [0] * self.r.size
        for i in range(0, self.r.size):
            dff[i] = (diff(self.f(), self.x[i]))
            replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
            df[i] = dff[i].subs(replacements)
        return np.array(df)

    def ineq_cons_1(self, sign):
        self.x = symbols('x[0:%d]' % self.r.size)
        icons1 = sign*(np.dot(self.x, self.r1) - self.r_star)
        return icons1

    def sub_ineq_cons_1(self,x,sign):
        self.x = symbols('x[0:%d]' % self.r.size)
        replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
        return self.ineq_cons_1(sign).subs(replacements)


    def diff_ineq_cons_1(self, x,sign):
        d_ineq_cons_1 = [0] * self.r.size
        dicons1 = [0] * self.r.size
        for i in range(0, self.r.size):
            d_ineq_cons_1[i] = (diff(self.ineq_cons_1(sign), self.x[i]))
            replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
            dicons1[i] = d_ineq_cons_1[i].subs(replacements)
        return  dicons1

    def ineq_cons_2(self,sign):
        self.x = symbols('x[0:%d]' % self.r.size)
        icons2 = sign*((np.dot(self.x, self.r)) - self.r_star)
        return icons2

    def sub_ineq_cons_2(self,x,sign):
        self.x = symbols('x[0:%d]' % self.r.size)
        replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
        return self.ineq_cons_2(sign).subs(replacements)



    def diff_ineq_cons_2(self, x, sign):
        d_ineq_cons_2 = [0] * self.r.size
        dicons2 = [0] * self.r.size
        for i in range(0, self.r.size):
            d_ineq_cons_2[i] = (diff(self.ineq_cons_2(sign), self.x[i]))
            replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
            dicons2[i] = d_ineq_cons_2[i].subs(replacements)
        return dicons2

    def eq_cons_2(self):
        self.x = symbols('x[0:%d]' % self.r.size)
        econs2 = sum(self.x) - 1
        return  econs2

    def sub_eq_cons_2(self,x):
        self.x = symbols('x[0:%d]' % self.r.size)
        replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
        return self.eq_cons_2().subs(replacements)

    def diff_eq_cons_2(self, x):
        d_eq_cons_2 = [0] * self.r.size
        decons2 = [0] * self.r.size
        for i in range(0, self.r.size):
            d_eq_cons_2[i] = (diff(self.eq_cons_2(), self.x[i]))
            replacements = [(self.x[i], x[i]) for i in range(self.r.size)]
            decons2[i] = d_eq_cons_2[i].subs(replacements)
        return decons2

    def calculate_X0(self):
        d = 0
        D = 0
        for i in range(0, self.r.size):
                if self.r[i] < self.r_star:
                    D = i
                    #print (self.r[D])
                if self.r[i] > self.r_star:
                    d = i
                    #print(self.r[d])
        r1 = self.r[d]
        r2 = self.r[D]
        r11 = self.r1[d]
        r12 = self.r1[D]
        r21 = self.r2[d]
        r22 = self.r2[D]
        x = symbols('x')
        F = (self.func_x0(x, r1, r2, r11, r12, r21, r22, self.bet))
        x0 = nsolve(F, 0.1, tol = 2.37661e-06)
        return ( 1-x0.real, x0.real, D, d)

    def calculate_X0_2(self):
        d = 0
        D = 0
        for i in range(0, self.r.size):
            if self.r[i] < self.r_star:
                d = i
            if self.r[i] > self.r_star:
                D = i
        r1 = self.r[d]
        #print(r1)
        r2 = self.r[D]
        #print(r2)
        r11 = self.r1[d]
        r12 = self.r1[D]
        r21 = self.r2[d]
        r22 = self.r2[D]
        x = symbols('x')
        F2 = (self.func_x0_2(x, r1, r2, r11, r12, r21, r22, self.bet))
        x0_2 = nsolve(F2, 0.1, tol = 9.94317e-03)
        return (1 - x0_2.real, x0_2.real, d, D)


    def x_betta(self):
        if self.bet < 0.52:
            x0 = [0] * self.r.size
            x0 = [0.2,0.1,0.7]
            # x0[self.calculate_X0()[2]] = self.calculate_X0()[0]
            # x0[self.calculate_X0()[3]] = self.calculate_X0()[1]
            print(x0)
            print(self.calculate_X0())
            cons = ({'type': 'eq',
                    'fun': lambda x: np.array([self.sub_betta(x)]),
                    'jac': lambda x: np.array([self.diff_betta(x)])},
                    {'type': 'eq',
                    'fun': lambda x: np.array([self.sub_eq_cons_2(x)]),
                    'jac': lambda x: np.array([self.diff_eq_cons_2(x)])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([self.sub_ineq_cons_1(x,1)]),
                    'jac': lambda x: np.array([self.diff_ineq_cons_1(x,1)])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([self.sub_ineq_cons_2(x,-1)]),
                    'jac': lambda x: np.array([self.diff_ineq_cons_2(x,-1)])},
                    {'type': 'ineq',
                     'fun': lambda x: np.array([-x[0]]),
                     'jac': lambda x: np.array([-1, 0, 0])},
                    {'type': 'ineq',
                     'fun': lambda x: np.array([-x[1]]),
                     'jac': lambda x: np.array([0, -1, 0])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([-x[2]]),
                    'jac': lambda x: np.array([0, 0, -1])},
                   # {'type': 'ineq',
                    #'fun': lambda x: np.array([-x[3]]),
                    #'jac': lambda x: np.array([0, 0, 0, -1, 0])},
                    #{'type': 'ineq',
                   # 'fun': lambda x: np.array([-x[4]]),
                   # 'jac': lambda x: np.array([0, 0, 0, 0, -1])}
                    )


            res = minimize(self.sub_f, x0, jac=self.diff_f, constraints=cons, method='SLSQP', options={'ftol': 1e-4, 'disp': True, "eps":1e-4, "maxiter": 10000})
            return(res)
        else:
            x0 = [0]*self.r.size
            x0[self.calculate_X0_2()[3]] = self.calculate_X0_2()[0]
            x0[self.calculate_X0_2()[2]] = self.calculate_X0_2()[1]
            print(x0)
            print(self.calculate_X0_2())
            cons1 = ({'type': 'eq',
                    'fun': lambda x: np.array([self.sub_betta2(x)]),
                    'jac': lambda x: np.array([self.diff_betta2(x)])},
                    {'type': 'eq',
                    'fun': lambda x: np.array([self.sub_eq_cons_2(x)]),
                    'jac': lambda x: np.array([self.diff_eq_cons_2(x)])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([self.sub_ineq_cons_1(x, -1)]),
                    'jac': lambda x: np.array([self.diff_ineq_cons_1(x, -1)])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([self.sub_ineq_cons_2(x, 1)]),
                    'jac': lambda x: np.array([self.diff_ineq_cons_2(x, 1)])},
                    {'type': 'ineq',
                     'fun': lambda x: np.array([-x[0]]),
                     'jac': lambda x: np.array([-1, 0, 0])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([-1*x[1]]),
                    'jac': lambda x: np.array([0, -1, 0])},
                    {'type': 'ineq',
                    'fun': lambda x: np.array([-1*x[2]]),
                    'jac': lambda x: np.array([0, 0, -1])},
                    #{'type': 'ineq',
                    #'fun': lambda x: np.array([-1*x[3]]),
                    #'jac': lambda x: np.array([0, 0, 0, -1, 0])},
                   #{'type': 'ineq',
                    #'fun': lambda x: np.array([-1*x[4]]),
                    #'jac': lambda x: np.array([0, 0, 0, 0, -1])}
            )



            res1 = minimize(self.sub_f, x0, jac=self.diff_f, constraints=cons1, method='SLSQP',
                       options={'ftol': 1e-4, 'disp': True, "eps": 1e-1, "maxiter": 10000})
            return(res1)

start = time.time()
portfolio = Portfolio([0, 0, 0], [r_KO, r_APL, r_MSFT], [r1_KO, r1_APL, r1_MSFT], [r2_KO, r2_APL, r2_MSFT], 0.015, 0.3)
print(portfolio.calculate_xi())
print("Finished in " + str(time.time() - start) + "sec")

#x = [0]*5
#res = [0]*5
#for i in range(0, 5, 0.1):
   # x[i] = 0.5 + i*0.1
    #portfolio = Portfolio([0, 0], [0.021, 0.048], [-0.01, -0.041], [0.039, 0.057], 0.035, x[i])
    #res[i] = -1*portfolio.x_betta().fun
#print(x, res)
#plt.plot(x, res)
#plt.show()
#print(portfolio.find_betta2(x))
#print(portfolio.f())

#print(portfolio.betta())
#print(portfolio.calculate_X0())
#print(portfolio.calculate_X0_2())
#print(portfolio.betta())


