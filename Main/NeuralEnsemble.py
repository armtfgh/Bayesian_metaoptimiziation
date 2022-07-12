
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.neural_network import MLPRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class NeuralEnsembleRegressor(BaseEstimator, RegressorMixin):

    """ 
    This is a regressor built on the scikit-learn template estimator.
    The central piece is sklearn.base.BaseEstimator. 
    In addition, scikit-learn provides a mixin, i.e. sklearn.base.RegressorMixin, 
    which implements the score method which computes the R^2 score of the predictions.

    Scikit-learn implements the neural network regressor via MLPRegressor.
    This implementation extends that to allow for computation of an ensemble of NNs.
    It takes the most important parameters of MLPRegressor, such as:
    hidden_layer_sizes, activation, solver, and max_iter.
    In addition, it accepts a new parameter: ensemble_size = N.
    The regressor thus creates N instances of the MLPRegressor.
    When predict() is called, the mean of the individual predictions is returned.
    In addition, when return_std=True, it returns the std of the predictions.
    This allows the regressor to be used with scikit-optimize. 
    """
    
    def __init__(self, hidden_layer_sizes=(120,130,120), activation='relu', solver='adam', 
                 max_iter=3000, tol=1e-4, ensemble_size=10):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.max_iter = max_iter
        self.ensemble_size = ensemble_size
        self.tol = tol

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self._regressor_array = []
        for i in range(self.ensemble_size):
            regressor = MLPRegressor(hidden_layer_sizes=self.hidden_layer_sizes, 
                                     activation=self.activation, 
                                     solver=self.solver, 
                                     max_iter=self.max_iter,
                                     tol=self.tol)
            regressor.fit(X,y)
            self._regressor_array.append(regressor)
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        predictions = []
        regressor_array = self._regressor_array
        for i in range(self.ensemble_size):
            prediction = regressor_array[i].predict(X)
            predictions.append(prediction)
        mat = np.matrix(predictions)
        if return_std:
            return np.array(mat.mean(0).tolist()[0]), np.array(mat.std(0).tolist()[0])
        else:
            return np.array(mat.mean(0).tolist()[0])



