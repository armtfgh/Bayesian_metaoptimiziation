
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from keras.layers import Dense, Dropout, Input
from keras.models import Model
from tensorflow.keras.optimizers import RMSprop


class NeuralDropoutEnsembleRegressor(BaseEstimator, RegressorMixin):

    """ 
    This is a regressor built on the scikit-learn template estimator.
    The central piece is sklearn.base.BaseEstimator. 
    In addition, scikit-learn provides a mixin, i.e. sklearn.base.RegressorMixin, 
    which implements the score method which computes the R^2 score of the predictions.

    A neural network regressor is implemented via Keras Dense NN, with Dropout. 
    In addition, it accepts a new parameter: ensemble_size = N.
    When predict() is called, N distinct predictions are generated.
    The mean of the individual predictions is returned.
    In addition, when return_std=True, it returns the std of the predictions.
    This allows the regressor to be used with scikit-optimize. 
    """
    
    def __init__(self, ensemble_size=10):
        self.ensemble_size = ensemble_size

        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        num_variables = X.shape[1]

        inputs = Input(shape=(num_variables,))
        x = inputs
        x = Dense(100, activation='relu', kernel_initializer='he_normal', 
                    input_shape=(num_variables,))(x)
        x = Dropout(rate=0.2)(x, training=True)
        x = Dense(100, activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(rate=0.2)(x, training=True)
        output = Dense(1)(x)

        model = Model(inputs,output)
        model.compile(optimizer=RMSprop(lr=0.01), loss='mse')
        model.fit(X, y, epochs=2000, verbose=0)
        
        self._regressor = model
        self.is_fitted_ = True
        return self

    def predict(self, X, return_std=False):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        predictions = []
        regressor = self._regressor
        for i in range(self.ensemble_size):
            prediction = regressor.predict(X)
            predictions.append(np.concatenate(prediction))
        mat = np.matrix(predictions)

        if return_std:
            return np.array(mat.mean(0).tolist()[0]), np.array(mat.std(0).tolist()[0])
        else:
            return np.array(mat.mean(0).tolist()[0])



