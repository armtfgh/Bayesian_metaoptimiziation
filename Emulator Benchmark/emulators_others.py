import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from xgboost import XGBRegressor
from numpy import absolute,std, mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from NeuralEnsemble import NeuralEnsembleRegressor
import sys
import warnings
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from olympus import Dataset
from olympus import Emulator
from olympus.models import BayesNeuralNet,NeuralNet

if not sys.warnoptions:
    warnings.simplefilter("ignore")


class emulators():
    def __init__(self, X,y,scale=False):
        self.x = X
        self.y =y
        self.scale = scale 
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y)
        self.poly_reg= PolynomialFeatures(degree=3)
        # self.x_poly_train = self.poly_reg.fit_transform(self.x_train)
        # self.x_poly_test = self.poly_reg.transform(self.x_test)
        self.df = df        
        if scale:
            scaler = StandardScaler()
            self.x_train = scaler.fit_transform(self.x_train, self.y_train)
            self.x_test = scaler.transform(self.x_test)
            
    def linreg(self):
        regressor=LinearRegression(normalize=False)
        return regressor 
    
    def sgd(self):
        regressor = SGDRegressor()
        return regressor 
    
    def svr(self):
        regressor = SVR(kernel='linear')
        return regressor 
    
    def decisiontree(self):
        regressor = DecisionTreeRegressor(random_state=1)
        return regressor 
    
    def randomforest(self):
        regressor = RandomForestRegressor(n_estimators=300,random_state=0)
        return regressor 
    
    def xgboost(self):
        regressor = XGBRegressor(alpha=0.3)
        return regressor 
    
    def ridge(self):
        regressor=Ridge(alpha=1.0)
        return regressor 
    
    def elas_net(self):
        regressor = ElasticNet(alpha=1)
        return regressor 
    
    def knn(self):
        regressor=KNeighborsRegressor()
        return regressor 
    
    def mlp(self,hidden_layers=(120,65,120)):
        regressor=MLPRegressor(max_iter=500,solver='adam',learning_rate='invscaling',activation='relu',hidden_layer_sizes=hidden_layers)
        return regressor 
    
    def gpr(self):
        kernel = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(
            length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
        regressor = GaussianProcessRegressor(kernel=kernel)
        return regressor 
    
    def nne(self):
        regressor = NeuralEnsembleRegressor(hidden_layer_sizes=(120,130,120), activation='relu', solver='adam', 
                 max_iter=3000, tol=1e-4, ensemble_size=10)
        return regressor 
    
    def adaboost(self):
        regressor = AdaBoostRegressor(n_estimators=100)
        return regressor 
    

    
    def evaluator(self):
        models = [self.linreg(),self.svr(),self.decisiontree(),
                  self.randomforest(),self.xgboost(),self.ridge(),self.elas_net(),self.knn(),
                  self.mlp(),self.nne(),self.adaboost()]
        
        R2,MAE,RMSE = [],[],[]
        for model in models:
            model.fit(self.x_train,self.y_train)
            y_pred= model.predict(self.x_test)
            R2.append(r2_score(self.y_test, y_pred))
            MAE.append(mean_absolute_error(self.y_test, y_pred))
            RMSE.append(mean_squared_error(self.y_test,y_pred)**0.5)
            
        
        names = ["Linear Regression","SVR","Decision Tree","Random Forest"
                 ,"XGBoost","Ridge","Elastic Network","KNN","NN","NNE","Adabosot"]
        
        
        df = pd.DataFrame(list(zip(names, R2,MAE,RMSE)),
                       columns =['Model Name', 'R2 Score','MAE','RMSE'])
        
        import plotly.io as pio
        pio.renderers.default='browser'
        import plotly.express as px



        fig = px.bar(df, x="Model Name", y="RMSE", )
        fig.update_layout(
            title={
                'text': "Machine Learning Models Performance,Mean Squared Error",
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

        
        
        
        
        
        
    
names = ["Linear Regression","SVR","Decision Tree","Random Forest"
          ,"XGBoost","Ridge","Elastic Network","KNN","NN","NNE"]        
        
n_ensembles = 10
df = pd.read_csv('data/benz.csv')
X = df.iloc[:,:-1]
y = df['yld'].to_numpy()
R2_ens = np.zeros((10,n_ensembles))
mae_ens = np.zeros((10,n_ensembles))
rmse_ens = np.zeros((10  ,n_ensembles))

for i in range(n_ensembles):

    # df1 = X.join(df['Yield'])
    emul = emulators(X, y,scale=True)
    
    
    emul.evaluator()
    
    models = [emul.linreg(),emul.svr(),emul.decisiontree(),
              emul.randomforest(),emul.xgboost(),emul.ridge(),emul.elas_net(),emul.knn(),
              emul.mlp(),emul.nne()]
    R2,MAE,RMSE = [],[],[]
    for model in models:
        model.fit(emul.x_train,emul.y_train)
        y_pred= model.predict(emul.x_test)
        R2.append(r2_score(emul.y_test, y_pred))
        MAE.append(mean_absolute_error(emul.y_test, y_pred))
        RMSE.append(mean_squared_error(emul.y_test,y_pred)**0.5)
    R2_ens[:,i] = np.copy(R2)
    mae_ens[:,i] = np.copy(MAE)
    rmse_ens[:,i] = np.copy(RMSE)

R2_avg = [np.average(R2_ens[i,:]) for i in range(10)]
R2_std = [np.std(R2_ens[i,:]) for i in range(10)]
mae_avg = [np.average(mae_ens[i,:]) for i in range(10)]
mae_std = [np.std(mae_ens[i,:]) for i in range(10)]
rmse_avg = [np.average(rmse_ens[i,:]) for i in range(10)]
rmse_std = [np.std(rmse_ens[i,:]) for i in range(10)]
    
    
df2 = pd.DataFrame(list(zip(names, R2_avg,R2_std,mae_avg,mae_std,rmse_avg,rmse_std)),
                columns =['Model Name', 'R2 Score','r2 error','MAE','maeerror','RMSE','rmseerror'])
    # plt.bar(names,R2)
    # import plotly.io as pio
    # pio.renderers.default='browser'
    # import plotly.express as px
    
    
    # fig = px.bar(df2, x="Model Name", y="R2 Score", )
    # fig.update_layout(
    #     title={
    #         'text': "Machine Learning Models Performance ",
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'})
    
    
            
# df2.to_csv("benz.csv")   
            
        
    
    
    
        