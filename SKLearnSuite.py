## Machine Learning Algorithm for Process Predictions

## These should be the primary focus
from sklearn.svm import SVR,LinearSVR,NuSVR
from sklearn.linear_model import Ridge as Ri
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR

## Using these to minimize number of variables
from sklearn.linear_model import Lasso,ElasticNet

## These will be experimental
from sklearn.neighbors import KNeighborsRegressor as KNNR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.isotonic import IsotonicRegression as IR

## Support Modules
from scipy import polyfit,polyval,stats
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import numpy as np
import sympy as sp
style.use('fivethirtyeight')

## Built-in Support Modules
import itertools as ITs
import functools as FTs
import random as rd
import math as mt
import os,sys

## Primary Machine Learning Alternatives
class DataEvaluation():

    def __init__(self,pd_DF,DVCols = [],IVCols = [],*DVdf,**IVdf):
        self.pd_DF = pd.read_excel(pd_DF) if type(pd_DF) == str else pd_DF
        self.DVCols = DVCols
        self.IVCols = IVCols
        self.TupleX = self.pd_DF[IVCols] if IVCols else 0
        self.X = [tuple(u) for u in self.TupleX.values]
        self.y = self.pd_DF[DVCols]
        self.DVdf = DVdf
        self.IVdf = IVdf

    def SupportVector(self,Results='',TestSet=False):
        '''Results==True: provides results for the model'''
        SV = SVR(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1)
        if TestSet==False:
            SVResult = SV.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(SVResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(SVResult.get_params()))
            plt.plot(SV.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            SVResult = SV.fit(x_train,y_train)
            if Results==True:
                print(str(SVResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(SVResult.get_params()))
            SVRPredict = SVResult.predict(x_test)
            plt.plot(SVRPredict,polyval(polyfit(SVRPredict,y_test.reshape(-1),1),SVRPredict),'r-',label='predicted')
            plt.plot(SVRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
        
    def LinearSupportVector(self,Results='',TestSet=False):
        LSV = LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000)
        if TestSet==False:
            LSVResult = LSV.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(LSVResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(LSVResult.get_params()))
            plt.plot(LSV.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            LSVResult = LSV.fit(x_train,y_train)
            if Results==True:
                print(str(LSVResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(LSVResult.get_params()))
            LSVRPredict = LSVResult.predict(x_test)
            plt.plot(LSVRPredict,polyval(polyfit(LSVRPredict,y_test.reshape(-1),1),LSVRPredict),'r-',label='predicted')
            plt.plot(LSVRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
                 
    def NuSupportVector(self,Results='',TestSet=False):
        NSV = NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, tol=0.001, cache_size=200, verbose=False, max_iter=-1)
        if TestSet==False:
            NSVResult = NSV.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(NSVResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(NSVResult.get_params()))
            plt.plot(NSV.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            NSVResult = NSV.fit(x_train,y_train)
            if Results==True:
                print(str(NSVResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(NSVResult.get_params()))
            NSVRPredict = NSVResult.predict(x_test)
            plt.plot(NSVRPredict,polyval(polyfit(NSVRPredict,y_test.reshape(-1),1),NSVRPredict),'r-',label='predicted')
            plt.plot(NSVRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
            
    def Ridge(self,Results='',TestSet=False):
        R = Ri(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
        if TestSet==False:
            RResult = R.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(RResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(RResult.get_params()))
            plt.plot(R.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            RResult = R.fit(x_train,y_train)
            if Results==True:
                print(str(RResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(RResult.get_params()))
            RRPredict = RResult.predict(x_test)
            plt.plot(RRPredict,polyval(polyfit(RRPredict,y_test.reshape(-1),1),RRPredict),'r-',label='predicted')
            plt.plot(RRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
            
    def Gradient(self,Results='',TestSet=False):
        G = GBR(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=None, init=None, random_state=None,
                max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
        if TestSet==False:
            GResult = G.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(GResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(GResult.get_params()))
            plt.plot(G.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            GResult = G.fit(x_train,y_train)
            if Results==True:
                print(str(GResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(GResult.get_params()))
            GRPredict = GResult.predict(x_test)
            plt.plot(GRPredict,polyval(polyfit(GRPredict,y_test.reshape(-1),1),GRPredict),'r-',label='predicted')
            plt.plot(GRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
            
    def Forest(self,Results='',TestSet=False):
        F = RFR(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
             max_leaf_nodes=None, min_impurity_split=.01, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
        if TestSet==False:
            FResult = F.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(FResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(FResult.get_params()))
            plt.plot(F.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            FResult = F.fit(x_train,y_train)
            if Results==True:
                print(str(FResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(FResult.get_params()))
            FRPredict = FResult.predict(x_test)
            plt.plot(FRPredict,polyval(polyfit(FRPredict,y_test.reshape(-1),1),FRPredict),'r-',label='predicted')
            plt.plot(FRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
            
## Class for lowest variables
class FewFeatures():

    def __init__(self,pd_SampleData,DVCols,IVCols):
        self.pd_SampleData = pd_SampleData
        self.DVCols = DVCols
        self.IVCols = IVCols

    def lasso(self):
        df = self.pd_SampleData
        la = Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
                   normalize=False, positive=False, precompute=False, random_state=None,
                   selection='cyclic', tol=0.0001, warm_start=False)
        IVdf = list(zip(df.Col1,df.Col2))
        DVdf = df.Col3.tolist()
        las = la.fit(IVdf,DVdf)
        print(str(la.coef_) + '\n' + str(la.score(IVdf,DVdf)))
        return None

    def elasticNet(self):
        pass

## Experimental Machine Learning
class Experimental():

    def __init__(self,pd_DF,DVCols = [],IVCols = []):
        self.pd_DF = pd.read_excel(pd_DF)
        self.DVCols = DVCols
        self.IVCols = IVCols
        self.TupleX = self.pd_DF[IVCols] if IVCols else 0
        self.X = [tuple(u) for u in self.TupleX.values]
        self.y = self.pd_DF[DVCols]

    def Neighbors(self,Results=True,TestSet=False):
        KNN = KNNR(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
        if TestSet==False:
            KNNResult = KNN.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(KNNResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(KNNResult.get_params()))
            plt.plot(KNN.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            KNNResult = KNN.fit(x_train,y_train)
            if Results==True:
                print(str(KNNResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(KNNResult.get_params()))
            KNNRPredict = KNNResult.predict(x_test)
            plt.plot(KNNRPredict,polyval(polyfit(KNNRPredict,y_test.reshape(-1),1),KNNRPredict),'r-',label='predicted')
            plt.plot(KNNRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
        

    def Neural(self,Results=True,TestSet=False):
        NN = MLPR(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                   power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                   early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        if TestSet==False:
            NNResult = NN.fit(self.X,np.ravel(self.y,1))
            if Results==True:
                print(str(NNResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(NNResult.get_params()))
            plt.plot(NN.fit(self.X,np.ravel(self.y,1)).predict(self.X))
            y = np.array(self.y[self.DVCols])
            plt.plot(y,'ro')
            plt.show()
        else:
            x_train = self.X[:len(self.X)//2]
            y_train = np.ravel(self.y,1)[:len(self.y)//2]
            x_test = self.X[len(self.X)//2:]
            y_test = np.ravel(self.y,1)[len(self.y)//2:]
            NNResult = NN.fit(x_train,y_train)
            if Results==True:
                print(str(NNResult.score(self.X,np.ravel(self.y,1))) + '\n' + str(NNResult.get_params()))
            NNRPredict = NNResult.predict(x_test)
            plt.plot(NNRPredict,polyval(polyfit(NNRPredict,y_test.reshape(-1),1),NNRPredict),'r-',label='predicted')
            plt.plot(NNRPredict,y_test.reshape(-1),'bo')
            plt.legend()
            plt.show()
                 
    def Iso(self):
        IReg = IR(y_min=None, y_max=None, increasing=True, out_of_bounds='nan')
        pass

def GraphIT():
    pass

## Multiple regression test vs machine Learning
def MultiReg(pdf,DV,IV):
    from statsmodels.formula.api import OLS,ols
    df = pd.read_excel(pdf)
    y,x1,x2 = df[DV[0]],df[IV[0]],df[IV[1]]
    print(len(x1))
    print(len(x2))
    x = np.column_stack((x1,x2))
    REG = OLS(y,x).fit()
    reg = ols(formula='y~x1+x2',data=df).fit()
    print(REG.summary())
    print('_'*50)
    print(reg.summary())
    eq1,eq2 = list(REG.params),list(reg.params)
    df['REG'] = eq1[0]*x1 + eq2[1]*x2
    df['reg'] = eq2[0] + eq2[1]*x1 + eq2[2]*x1
    plt.plot(df['REG'],'b-')
    plt.plot(y,'rs')
    plt.show()
    plt.clf()
    plt.plot(df['reg'],'b-')
    plt.plot(y,'rs')
    plt.show()

##############TEST_CASES##################
##EE = Experimental('TestDataFrame2.xlsx',['Col1'],['Col2','Col3'])
##EE.Neighbors()
    #
##MultiReg('TestDataFrame2.xlsx',['Col1'],['Col2','Col3'])
    #
##test = FewFeatures(pd.read_excel('TestDataFrame1.xlsx'),0,0)
##test.lasso()
    #
##DE = DataEvaluation('TestDataFrame2.xlsx',['Col1'],['Col2','Col3'])
##DE.Forest(Results=True,TestSet=False)
