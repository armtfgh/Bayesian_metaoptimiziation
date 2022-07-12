from skopt.benchmarks import branin as _branin
import numpy as np
np.random.seed(123)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from NeuralEnsemble import NeuralEnsembleRegressor
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from matplotlib.colors import LogNorm
from skopt import gp_minimize
from skopt.optimizer import Optimizer
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import random
from sklearn.linear_model import LinearRegression


# main class for meta optimizer
class dynamicsurrogate:
    def __init__(self,X,y,df,n_initial_points,initial_point_generator,random_state):
        self.X = X
        self.y = y
        self.df = df
        # self.regressor = XGBRegressor(alpha=0.4)
        self.regressor = NeuralEnsembleRegressor(ensemble_size=10)

        self.regressor.fit(X,y)
        self.columns = self.df.columns
        self.n_initial_points=n_initial_points
        self.initial_point_generator = initial_point_generator
        self.random_state = random_state
        self.Y=[]
        self.Xs = []
        self.acq_func = "EI"
        
        
        
        
        n_dims = len(self.columns)-1
        self.min_ = []
        self.max_ = []
        
        for i in range(len(self.columns)-1):
            self.min_.append(self.df.iloc[:,i].min())
            self.max_.append(self.df.iloc[:,i].max())
        self.bounds = []
        
        for i in range(len(self.columns)-1):
            self.bounds.append((self.min_[i],self.max_[i]))
        #matern kernel for GP optimizer
        self.matern_fixed = ConstantKernel(1.0, constant_value_bounds='fixed') * Matern(
            length_scale=np.ones(n_dims), length_scale_bounds='fixed', nu=2.5)
        self.matern_tunable = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
            
        #defining regressors used within the optimizers
        self.gpr = GaussianProcessRegressor(kernel= None) 
        self.ner = NeuralEnsembleRegressor(ensemble_size=10)
        
        #setting up the optimizers
        self.opter1 = self.opt_gen(self.gpr)
        self.opter2 = Optimizer(self.bounds,base_estimator=self.ner,n_initial_points=self.n_initial_points,initial_point_generator=self.initial_point_generator,random_state=self.random_state,acq_optimizer="sampling")
        self.opter3 = self.opt_gen('rf')
        self.opter4 = self.opt_gen('gbrt')
        # self.opter5 = self.opt_gen('et')   #additional optimizer to be added
        # self.opters = [self.opter1,self.opter2,self.opter3,self.opter4,self.opter5]
        
        

        
        
        
        self.opters = [self.opter1,self.opter2,self.opter3,self.opter4]
        
        #defining different type of regresssor for construction of predictive emulator
        self.inner_models = [XGBRegressor(alpha=0.45,gamma=0.1),KNeighborsRegressor(n_neighbors = n_initial_points),NeuralEnsembleRegressor(hidden_layer_sizes=(120), activation='relu', solver='adam', 
                     max_iter=2000, tol=1e-4, ensemble_size=10),LinearRegression(),GaussianProcessRegressor(self.matern_tunable),RandomForestRegressor()]
        self.regressor_list = [self.gpr,self.ner,RandomForestRegressor(),GradientBoostingRegressor()]
        
        self.sug_indexes = []
        self.real_indexes=[]
        self.sugs=[]
        self.sug_vals = []
        self.real_sug_vals = []
        
        self.gprX,self.nerX,self.rfX,self.gbrtX,self.etrX = [],[],[],[],[]
        self.gpMAE,self.nerMAE,self.rfMAE,self.gbrtMAE,self.etMAE = [],[],[],[],[],
        self.gpMSE,self.nerMSE,self.rfMSE,self.gbrtMSE,self.etMSE = [],[],[],[],[],
        self.gpr2,self.nerr2,self.rfr2,self.gbrtr2,self.etr2 = [],[],[],[],[],
        self.Preds =[]
        
        
        
    def obj_func(self,x):
        x = np.array(x)
        x = x.reshape(-1,len(self.columns)-1)
        pred = self.regressor.predict(x)
        return -pred[0]
     
    def opt_gen(self,base_estimator):
        optimizer = Optimizer(self.bounds,base_estimator=base_estimator,n_initial_points=self.n_initial_points,initial_point_generator=self.initial_point_generator,random_state=self.random_state,acq_func=self.acq_func)
        return optimizer
    
    def tell_all(self,x,y):
        for objects in self.opters:
            objects.tell(x,y)
    
    def ask_all(self):
        suggestions = []
        for objects in self.opters:
            suggestions.append(objects.ask())
        return suggestions
    
    def min_ind_finder(self,arr):
        ind= min(enumerate(arr), key=(lambda x: x[1]))
        return ind[0]
    
    def min_ind_finder_from_one(self,arr):
        vals= []
        for i in arr:
            vals.append(np.abs(i-1))
        
        ind= min(enumerate(vals), key=(lambda x: x[1]))
        return ind[0]
    
    
    
    def opt_selector(self,regressor,points):
        results = []
        for i in points:
            results.append(regressor.predict(np.asarray(i).reshape(1,-1)))
        
        min_ind = self.min_ind_finder(results)
        return points[min_ind],results
    
    def random_generator(self,a,b):
        return (b-a)*np.random.rand()+a
    
    def random_sample_selector(self):
        X1 = self.random_generator(self.min_[0],self.max_[0])
        X3 = self.random_generator(self.min_[1],self.max_[1])
        X4 = self.random_generator(self.min_[2],self.max_[2])
        X5 = self.random_generator(self.min_[3],self.max_[3])
        X6 = self.random_generator(self.min_[4],self.max_[4])
        X7 = self.random_generator(self.min_[5],self.max_[5])

        array = [X1,X3,X4,X5,X6,X7]
        return np.asarray(array)
    
    def fitter_predictor(self,regressor,X,y,suggested_point):
        predictions=[]
        regressor.fit(X,y)
        pred=regressor.predict(np.asarray(suggested_point).reshape(1,-1))
        return pred
    
    def distance_evaluator(self,array,y):
        out = []
        for vals in array:
            out.append(np.abs(vals-y))
        min_ind = self.min_ind_finder(out)
        return min_ind
            
    def ratio_evaluator(self,array,y):
        out = []
        for vals in array:
            out.append(np.abs(vals/y))
        min_ind = self.min_ind_finder_from_one(out)
        return min_ind
    
    def mae_func(self,regressor,X,y):
        # X = np.asanyarray(X).reshape(1,-1)
        regressor.fit(X,y)
        y_pred = regressor.predict(X)
        mae = mean_absolute_error(y,y_pred)
        return mae
    
    def mae_func_gp(self,regressor,X,y):
        # X = np.asanyarray(X).reshape(1,-1)
        y_pred = regressor.predict(X)
        mae = mean_absolute_error(y,y_pred)
        return mae
    
    def mse_func(self,regressor,X,y):
        # X = np.asanyarray(X).reshape(1,-1)
        regressor.fit(X,y)
        y_pred = regressor.predict(X)
        mse = mean_squared_error(y,y_pred)
        return mse
    
    def r2_func(self,regressor,X,y):
        # X = np.asanyarray(X).reshape(1,-1)
        regressor.fit(X,y)
        y_pred = regressor.predict(X)
        r2 = r2_score(y,y_pred)
        return r2
            
        
    

    
#meta optimizer core calcuation function based on ask and tell
    def optimize(self,n):
        for j in range(n):
            if j<self.n_initial_points:
                gprx = self.opters[1].ask()
                yyy = self.obj_func(gprx)
                self.tell_all(gprx,yyy)
                self.Y.append(yyy)
                self.Xs.append(gprx)
            
            else:
                self.predictions = []
                self.real_predictions = []
                sug_points =[]
                for i in range(len(self.opters)):
                    suggested_point=self.opters[i].ask()
                    self.real_predictions.append(self.obj_func(suggested_point))
                    sug_points.append(suggested_point)
                    value=self.fitter_predictor(self.regressor_list[i], self.Xs, self.Y, suggested_point)
                    self.predictions.append(value)
                self.predictions = np.asarray(self.predictions)
                self.real_predictions = np.asarray(self.real_predictions)
                min_ind = np.argmax(-self.predictions)
                real_min_ind = np.argmax(-self.real_predictions)
                
                self.Preds.append(self.predictions)
                # min_ind = self.distance_evaluator(self.predictions,self.Y[-1])
                # min_ind = self.ratio_evaluator(self.predictions,self.Y[-1])
                self.sug_indexes.append(min_ind)
                self.real_indexes.append(real_min_ind)
                next_suggestion =sug_points[min_ind]
                next_point_value = self.obj_func(next_suggestion)
                self.Y.append(next_point_value)
                self.Xs.append(next_suggestion)
                self.tell_all(next_suggestion,next_point_value)
        
    def optimize2(self,n):
        for j in range(n):
            if j<self.n_initial_points:
                gprx = self.opters[2].ask()
                yyy = self.obj_func(gprx)
                self.tell_all(gprx,yyy)
                self.Y.append(yyy)
                self.Xs.append(gprx)
            
            else:
                self.predictions = []
                sug_points =[]
                for i in range(len(self.opters)):
                    suggested_point=self.opter[i].ask()
                    sug_points.append(suggested_point)
                    self.predictions.append(self.fitter_predictor(self.regressor_list[i], self.Xs, self.Y, suggested_point))
                min_val = min(np.asarray(self.predictions)**7)
                min_ind = (np.asarray(self.predictions)**7).index(min_val)
                
                # self.Preds.append(self.predictions)
                # min_ind = self.distance_evaluator(self.predictions,self.Y[-1])
                # min_ind = self.ratio_evaluator(self.predictions,self.Y[-1])
                self.indexes.append(min_ind)
                next_suggestion = self.opters[min_ind].ask()
                next_point_value = self.obj_func(next_suggestion)
                self.Y.append(next_point_value)
                self.Xs.append(next_suggestion)
                self.tell_all(next_suggestion,next_point_value) 

                
#regressor based optimizer                
    def optimize_random(self,n):
        Y=[]
        X = []
        for j in range(n):
            if j<self.n_initial_points:
                X.append(self.random_sample_selector())
                Y.append(self.obj_func(X[-1]))
            else:
                regressor = self.inner_models[0]
                regressor.fit(np.asarray(X),np.asanyarray(Y))
                x_sugs = np.asarray([self.random_sample_selector() for i in range(20)])
                y_sugs = regressor.predict(x_sugs)
                X.append(x_sugs[self.min_ind_finder(y_sugs)])
                Y.append(self.obj_func(X[-1]))
        return X, Y
    
    
def accuracy_checker(arr1,arr2):
    res = []
    for i in range(len(arr1)):
        if arr1[i]==arr2[i]:
            res.append(1)
        else:
            res.append(0)
    return np.sum(res)/int(len(arr1))
        
                


def comparor(arr1,arr2):
    ind1 = min(enumerate(arr1), key=(lambda x: x[1]))[0]
    ind2 = min(enumerate(arr2), key=(lambda x: x[1]))[0]
    if ind1==ind2:
        return 1
    else:
        return 0
    
def array_comparor(res1,res2):
    out = []
    for i in range(len(res1)):
        result = comparor(res1[i], res2[i])
        out.append(result)
    return out,np.sum(out)/int(len(res1))

import pandas as pd 
random.seed(1231)



#initializing the benchmarking process
n_initial_points=3
initial_point_generator= 'random'
random_state =6
dataframe = pd.read_csv('data/yield.csv')
X = dataframe.iloc[:,:-1].to_numpy()
y = dataframe.iloc[:,-1].to_numpy()
model = dynamicsurrogate(X,y,dataframe,n_initial_points,initial_point_generator,random_state) #meta optimizer
n=30
n_ensemble =10  
res = np.zeros((n,n_ensemble))
res2 = np.zeros((n,n_ensemble))
res3 = np.zeros((n,n_ensemble))
res4 = np.zeros((n,n_ensemble))
res5 = np.zeros((n,n_ensemble))
res6 = np.zeros((n,n_ensemble))

sug_indexes_ens = np.zeros((n-n_initial_points,n_ensemble))
real_indexes_ens = np.zeros((n-n_initial_points,n_ensemble))
sum_ens = []

for i in range(n_ensemble):
    random_state=np.random.randint(0,341412)
    print("Ensemble :{}".format(i))
    # random_state=i
    model = dynamicsurrogate(X,y,dataframe,n_initial_points,initial_point_generator,random_state)
    model.optimize(n)
    sugsss =model.sugs
    sug_vals = model.sug_vals
    real_sug_vals = model.real_sug_vals
    Y = model.Y
    _ , Y_random = model.optimize_random(n)
    res[:,i]=Y
    opter_rf =Optimizer(model.bounds,base_estimator='rf',n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=model.acq_func)
    opter_gp=Optimizer(model.bounds,base_estimator=model.gpr,n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=model.acq_func)
    opter_gbr =Optimizer(model.bounds,base_estimator='gbrt',n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=model.acq_func)
    opter_ner =Optimizer(model.bounds,base_estimator=model.ner2,n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_optimizer="sampling",acq_func=model.acq_func)
    opter_rf.run(model.obj_func,n)
    opter_gp.run(model.obj_func,n)
    opter_gbr.run(model.obj_func,n) 
    opter_ner.run(model.obj_func,n)
    Y1_rf = opter_rf.yi
    Y1_gp = opter_gp.yi
    Y1_gbr = opter_gbr.yi 
    Y1_ner = opter_ner.yi 
    res2[:,i]=Y1_rf
    res3[:,i]=Y1_gp
    res4[:,i]=Y1_gbr
    res5[:,i] = Y_random
    res6[:,i]=Y1_ner
    sug_indexes_ens[:,i]= model.sug_indexes
    real_indexes_ens[:,i]= model.real_indexes
    summ = accuracy_checker(model.sug_indexes,model.real_indexes)
    print("the accuracy is: {}".format(summ)) 
    sum_ens.append(summ)
    

       
        
        
    
    
avg = []
for i in range(len(res[:,0])):
    avg.append(np.average(res[i,:]))
    

avg2 = []
for i in range(len(res2[:,0])):
    avg2.append(np.average(res2[i,:]))
    

avg3 = []
for i in range(len(res3[:,0])):
    avg3.append(np.average(res3[i,:]))

avg4 = []
for i in range(len(res4[:,0])):
    avg4.append(np.average(res4[i,:]))


avg5 = []
for i in range(len(res5[:,0])):
    avg5.append(np.average(res5[i,:]))

avg6 = []
for i in range(len(res6[:,0])):
    avg6.append(np.average(res6[i,:]))



plt.plot(range(n),-np.array(avg),label='New Model')
plt.plot(range(n),-np.array(avg2),label='Random Forrest')
plt.plot(range(n),-np.array(avg3),label='Gaussian Process')
plt.plot(range(n),-np.array(avg4),label='Gradient Boosting')
plt.plot(range(n),-np.array(avg5),label='Random XGB')
plt.plot(range(n),-np.array(avg6),label='NNE')
plt.legend()
plt.xlabel("Experiment Run")
plt.ylabel("Yield")


def array_modifier(array):
    new=[]
    ref = array[0]-1
    for i in array:
        if i<ref:
            ref = i
            new.append(i)
        else:
            new.append(ref)
    return new
            
avg_new = array_modifier(avg)
avg2_new = array_modifier(avg2)
avg3_new = array_modifier(avg3)
avg4_new = array_modifier(avg4)
avg5_new = array_modifier(avg5)
avg6_new = array_modifier(avg6)

        
plt.figure()        
plt.plot(range(n),-np.array(avg_new),label='New Model')
plt.plot(range(n),-np.array(avg2_new),label='Random Forrest')
plt.plot(range(n),-np.array(avg3_new),label='Gaussian Process')
plt.plot(range(n),-np.array(avg4_new),label='Gradient Boosting')
plt.plot(range(n),-np.array(avg5_new),label='Random XGB')
plt.plot(range(n),-np.array(avg6_new),label='NNE')
plt.legend()
plt.xlabel("Experiment Run")
plt.ylabel("Yield")
plt.title("init points= {}, ensemble size = {}, n = {},".format(n_initial_points,n_ensemble,n))


# a = np.random.randint(0,2142423232)
# plt.savefig(str(a)+".png")

# numpy_array = np.array(avg_new,avg_new,avg_new,avg_new)


#######################for dummy##############################################
def dummy_generator(n,n_ensemble):
    res = np.zeros((n,n_ensemble))          
    from skopt import dummy_minimize
    for kj in range(n_ensemble):
        opter_dummy = dummy_minimize(model.obj_func, model.bounds,n_calls=n)
        res[:,kj]=opter_dummy.func_vals
    res = array_changer(res)
    avg = array_averager(res)
    return res,avg



############ GP based benchmarking #############################################################################
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import ConstantKernel, Matern, DotProduct, WhiteKernel
from skopt import gp_minimize
def gp_matern_fixed(acq_func):
    matern_fixed = ConstantKernel(1.0, constant_value_bounds='fixed') * Matern(
        length_scale=np.ones(6), length_scale_bounds='fixed', nu=2.5)
    gpregressor = GaussianProcessRegressor(kernel=matern_fixed, n_restarts_optimizer=10, alpha=2, 
                                      normalize_y=True, noise='gaussian')
    
    # Perform the optimization with the GP regressor as the surrogate function
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state = kj
        resgp = gp_minimize(model.obj_func, 
                             model.bounds,
                             n_calls = n, 
                             base_estimator=gpregressor,
                             acq_func=acq_func, 
                             n_initial_points=n_initial_points,
                             initial_point_generator=initial_point_generator,
                             random_state=random_state,
                             )
        res[:,kj]=resgp.func_vals
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))

    avg_new = array_modifier(avg)
    return avg_new

def gp_matern_tunable(acq_func):
    matern_tunable = ConstantKernel(1.0, (1e-5, 1e5)) * Matern(length_scale=1.0, length_scale_bounds=(1e-5, 1e5), nu=2.5)
    gpregressor = GaussianProcessRegressor(kernel=matern_tunable, n_restarts_optimizer=10, alpha=2, 
                                      normalize_y=True, noise='gaussian')
    
    # Perform the optimization with the GP regressor as the surrogate function
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state=kj
        resgp = gp_minimize(model.obj_func, 
                             model.bounds,
                             n_calls = n, 
                             base_estimator=gpregressor,
                             acq_func=acq_func, 
                             n_initial_points=n_initial_points,
                             initial_point_generator=initial_point_generator,
                             random_state=random_state,
                             )
        res[:,kj]=resgp.func_vals
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new

def gp_rbf(acq_func):
    gpregressor = GaussianProcessRegressor(n_restarts_optimizer=10, alpha=2, 
                                      normalize_y=True, noise='gaussian')
    
    # Perform the optimization with the GP regressor as the surrogate function
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state=kj
        resgp = gp_minimize(model.obj_func, 
                             model.bounds,
                             n_calls = n, 
                             base_estimator='gp',
                             acq_func=acq_func, 
                             n_initial_points=n_initial_points,
                             initial_point_generator=initial_point_generator,
                             random_state=random_state,
                             )
        res[:,kj]=resgp.func_vals
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new

def gp_plot():
    matern_fix_EI = -np.asarray(gp_matern_fixed('EI'))
    matern_fix_PI = -np.asarray(gp_matern_fixed('PI'))
    # matern_fix_LCB = -np.asarray(gp_matern_fixed('LCB'))
    matern_tunable_EI = -np.asarray(gp_matern_tunable('EI'))
    matern_tunable_PI = -np.asarray(gp_matern_tunable('PI'))
    # matern_tunable_LCB = -np.asarray(gp_matern_tunable('LCB'))
    rbf_EI = -np.asarray(gp_rbf('EI'))
    rbf_PI = -np.asarray(gp_rbf('PI'))
    # rbf_LCB = -np.asarray(gp_rbf('LCB'))
    runs = list(range(n))
    # vals=[matern_fix_EI,matern_fix_PI,matern_fix_LCB,matern_tunable_EI,matern_tunable_PI,matern_tunable_LCB
    #       ,rbf_EI,rbf_PI,rbf_LCB]
    vals=[matern_fix_EI,matern_fix_PI,matern_tunable_EI,matern_tunable_PI
          ,rbf_EI,rbf_PI]
    
    # names = ['matern_fix_EI','matern_fix_PI','matern_fix_LCB','matern_tunable_EI','matern_tunable_PI','matern_tunable_LCB'
    #       ,'rbf_EI','rbf_PI','rbf_LCB']
    names = ['matern_fix_EI','matern_fix_PI','matern_tunable_EI','matern_tunable_PI'
          ,'rbf_EI','rbf_PI']
    for i in range(len(vals)):
        plt.plot(runs,vals[i],label=names[i])
    plt.legend()
    
def gp_data_extractor():
    matern_fix_EI = -np.asarray(gp_matern_fixed('EI'))
    matern_fix_PI = -np.asarray(gp_matern_fixed('PI'))
    matern_fix_gph = -np.asarray(gp_matern_fixed('gp_hedge'))
    matern_tunable_EI = -np.asarray(gp_matern_tunable('EI'))
    matern_tunable_PI = -np.asarray(gp_matern_tunable('PI'))
    matern_tunable_gph = -np.asarray(gp_matern_tunable('gp_hedge'))
    rbf_EI = -np.asarray(gp_rbf('EI'))
    rbf_PI = -np.asarray(gp_rbf('PI'))
    rbf_gph = -np.asarray(gp_rbf('gp_hedge'))
    
    dictiom = {'matern_fix_EI':matern_fix_EI,
                'matern_fix_PI':matern_fix_PI,
                'matern_tunable_EI':matern_tunable_EI,
                'matern_tunable_PI':matern_tunable_PI,
                'RBF_EI':rbf_EI,
                'RBF_PI':rbf_PI,
                'matern_fix_LCB':matern_fix_gph,
                'matern_tunable_LCB': matern_tunable_gph,
                'RBF_LCB':rbf_gph,}
    
    
    dataframe1 = pd.DataFrame(dictiom)
    
    dataframe1.to_csv('gp_based.csv')
    
############ Tree based benchmarking #############################################################################
from sklearn.ensemble import RandomForestRegressor

    
def rf(acq_func):
    # Perform the optimization with the GP regressor as the surrogate function
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state = kj
        opter_rf =Optimizer(model.bounds,base_estimator='rf',n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=acq_func)
        opter_rf.run(model.obj_func,n)
        res[:,kj]=opter_rf.yi
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new

def gbr(acq_func):
    # Perform the optimization with the GP regressor as the surrogate function
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state = kj
        opter_rf =Optimizer(model.bounds,base_estimator='gbrt',n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=acq_func)
        opter_rf.run(model.obj_func,n)
        res[:,kj]=opter_rf.yi
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new

def et(acq_func):
    # Perform the optimization with the GP regressor as the surrogate function
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state = kj
        opter_rf =Optimizer(model.bounds,base_estimator='et',n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=acq_func)
        opter_rf.run(model.obj_func,n)
        res[:,kj]=opter_rf.yi
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new


def tree_based_data_extractor():
    rf_EI = -np.asarray(rf('EI'))
    rf_PI = -np.asarray(rf('PI'))
    rf_gph = -np.asarray(rf('gp_hedge'))
    gbrt_EI = -np.asarray(gbr('EI'))
    gbrt_PI = -np.asarray(gbr('PI'))
    gbrt_gph = -np.asarray(gbr('gp_hedge'))
    et_EI = -np.asarray(et('EI'))
    et_PI = -np.asarray(et('PI'))
    et_gph = -np.asarray(et('gp_hedge'))
    
    dictiom = {'rf_EI':rf_EI,
                'rf_PI':rf_PI,
                'rf_gph':rf_gph,
                'gbrt_EI':gbrt_EI,
                'gbrt_PI':gbrt_PI,
                'gbrt_gph':gbrt_gph,
                'et_EI':et_EI,
                'et_PI': et_PI,
                'et_gph':et_gph,}
    
    
    dataframe1 = pd.DataFrame(dictiom)
    
    dataframe1.to_csv('tree_based.csv')




############ Network based benchmarking #############################################################################
from NeuralEnsembleDropout import NeuralDropoutEnsembleRegressor
from NeuralEnsemble import NeuralEnsembleRegressor

def nnDO(acq_func):
    keregressor = NeuralDropoutEnsembleRegressor(ensemble_size=50)
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state = kj
        opter_dro =Optimizer(model.bounds,base_estimator=keregressor,n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=acq_func,acq_optimizer="sampling")
        opter_dro.run(model.obj_func,n)
        res[:,kj]=opter_dro.yi
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new


def nne(acq_func):
    nner =NeuralEnsembleRegressor(ensemble_size=10)
    res = np.zeros((n,n_ensemble))  
    for kj in range(n_ensemble):
        random_state = kj
        opter_dro =Optimizer(model.bounds,base_estimator=nner,n_initial_points=n_initial_points,initial_point_generator=initial_point_generator,random_state=random_state,acq_func=acq_func,acq_optimizer="sampling")
        opter_dro.run(model.obj_func,n)
        res[:,kj]=opter_dro.yi
    avg = []
    for i in range(len(res[:,0])):
        avg.append(np.average(res[i,:]))
    avg_new = array_modifier(avg)
    return avg_new


def network_based_data_extractor():
    nne_EI = -np.asarray(rf('EI'))
    nne_PI = -np.asarray(rf('PI'))
    nne_gph = -np.asarray(rf('gp_hedge'))
    nnDO_EI = -np.asarray(gbr('EI'))
    nnDO_PI = -np.asarray(gbr('PI'))
    nnDO_gph = -np.asarray(gbr('gp_hedge'))

    dictiom = {'nne_EI':nne_EI,
                'nne_PI':nne_PI,
                'nne_gph':nne_gph,
                'nnDO_EI':nnDO_EI,
                'nnDO_PI':nnDO_PI,
                'nnDO_gph':nnDO_gph,}

    
    dataframe1 = pd.DataFrame(dictiom)
    dataframe1.to_csv('netowrk_based.csv')
import math 



def list_changer(array):
    new = []
    new.append(array[0])
    for i in np.arange(1,len(array)):
        if np.abs(array[i])>np.abs(new[-1]):
            new.append(array[i])
        else:
            new.append(new[-1])
    return new 



def normalizer(array):
    for i in range(len(array)):
        if array[i]<=-100:
            array[i]=np.copy(-100)
        elif array[i]>=0:
            array[i]=np.copy(0)
    return array    
            
        

def array_changer(res):
    res_output = np.zeros(res.shape)
    for i in range(len(res[0,:])):
        arr = normalizer(res[:,i])
        res_output[:,i]= list_changer(arr)
    return res_output
    
def array_averager(res):
    out = []
    for i in range(len(res[:,0])):
        out.append(np.average(res[i,:]))
    out = np.asarray(out)
    return out

def array_stder(res):
    out = []
    for i in range(len(res[:,0])):
        out.append(np.std(res[i,:]))
    out = np.asarray(out)
    return out
  
    
#exctracting the results gained from optimization  
def data_output(EI):
    pd.DataFrame(np.append(np.append(res,np.asarray(avg).reshape(-1,1),axis=1),np.asarray(avg_new).reshape(-1,1),axis=1)).to_csv(EI+"new_model.csv")
    pd.DataFrame(np.append(np.append(res2,np.asarray(avg2).reshape(-1,1),axis=1),np.asarray(avg2_new).reshape(-1,1),axis=1)).to_csv(EI+"rf.csv")
    pd.DataFrame(np.append(np.append(res3,np.asarray(avg3).reshape(-1,1),axis=1),np.asarray(avg3_new).reshape(-1,1),axis=1)).to_csv(EI+"gp.csv")
    pd.DataFrame(np.append(np.append(res4,np.asarray(avg4).reshape(-1,1),axis=1),np.asarray(avg4_new).reshape(-1,1),axis=1)).to_csv(EI+"gbr.csv")
    pd.DataFrame(np.append(np.append(res5,np.asarray(avg5).reshape(-1,1),axis=1),np.asarray(avg5_new).reshape(-1,1),axis=1)).to_csv(EI+"xrand.csv")
    pd.DataFrame(np.append(np.append(res6,np.asarray(avg6).reshape(-1,1),axis=1),np.asarray(avg6_new).reshape(-1,1),axis=1)).to_csv(EI+"nne.csv")

reses = [res,res2,res3,res4,res5,res6]
modified_reses = [array_changer(i) for i in reses]
averaged_vals = [array_averager(i) for i in modified_reses]
res_dummy,avg_dummy = dummy_generator(n, n_ensemble)
radar_vals = {}
def exporter(mod_res,res,name):
    averaged_vals = array_averager(mod_res)
    averaged_vals_res = array_averager(res)
    std = np.average(array_stder(res))
    dum_dist= np.average(averaged_vals-avg_dummy)
    max_val = averaged_vals[-1]
    average_val = np.average(averaged_vals_res)
    ef_avg = np.average(averaged_vals/avg_dummy)
    radar_vals[name] = [std,dum_dist,max_val,average_val,ef_avg]

names = ["New Model","RF","GP","GBR","Xrand","nne"]

for i in range(len(reses)):
    exporter(modified_reses[i],reses[i],names[i])
    
radar_graphs = pd.DataFrame(radar_vals)
    

    
def data_out():
    for i in range(len(reses)):
        pd.DataFrame(modified_reses[i]).to_csv(model.acq_func+" "+names[i]+".csv")
# data_out()
# radar_graphs.to_csv(model.acq_func+"radar.csv")
    
    
    
# data_output(model.acq_func)

