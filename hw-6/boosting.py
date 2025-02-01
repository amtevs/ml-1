from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional, Tuple


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds: int = None,
        eval_set: Optional[Tuple[np.ndarray]] = None,
        subsample=1.0, 
        bagging_temperature=1.0, 
        bootstrap_type='Bernoulli',
        goss: bool = False,
        goss_k: float = 0.2,
        rsm: float = 1.0,
        quantization_type = None,
        nbins: int = 255
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_set = eval_set
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.goss = goss
        self.goss_k = goss_k
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: y * self.sigmoid(-y * z)


    def partial_fit(self, X, y, resid):
        model = self.base_model_class(**self.base_model_params)
        index, X_train, y_train, weights_train = self.bootstrap(X, y, resid)
        model.fit(X_train, resid[index], sample_weight=weights_train)
        new_pred = model.predict(X)
        self.models.append(model)
        return new_pred
    
    def bootstrap(self, X, y, residuals):
        if self.goss:
            gradients = np.abs(residuals)
            n_large = int(len(gradients) * self.goss_k)
            large_index = np.argsort(gradients)[-n_large:]
            small_index = np.argsort(gradients)[:-n_large]

            n_small = int(len(small_index) * self.subsample)
            small_bootstrap_index = np.random.choice(small_index, n_small, replace=False)

            index = np.concatenate([large_index, small_bootstrap_index])

            weights = np.ones(len(y))
            weights[small_bootstrap_index] *= len(small_index) / n_small

        if self.bootstrap_type == 'Bernoulli':
                index = np.random.choice(X.shape[0], int(X.shape[0] * self.subsample), replace=True)
                weights = np.ones(sum(index))
        elif self.bootstrap_type == 'Bayesian':
                weights = (-np.log(np.random.rand(X.shape[0]))) ** self.bagging_temperature
                index = np.random.choice(np.arange(X.shape[0]), size=X.shape[0], p=weights / weights.sum())
        return index, X[index], y[index], weights[index]
    

    def quantize(self, X):
        if self.quantization_type == "Uniform":
            X = X.toarray()
            min_val, max_val = X.min(axis=0), X.max(axis=0)
            bins = np.array([np.linspace(min_val[i], max_val[i], self.nbins + 1) for i in range(X.shape[1])])
            X_quant = np.array([np.digitize(X[:, i], bins[i], right=False) for i in range(X.shape[1])]).T
        elif self.quantization_type == "Quantile":
            X = X.toarray()
            bins = np.array([np.percentile(X[:, i], np.linspace(0, 100, self.nbins)) for i in range(X.shape[1])])
            X_quant = np.array([np.digitize(X[:, i], bins[i], right=False) for i in range(X.shape[1])]).T
        #     min_val, max_val = X.min(axis=0), X.max(axis=0)
        #     bins = np.linspace(min_val, max_val, self.nbins + 1)
        #     X_quant = np.digitize(X, bins, right=False)
        # elif self.quantization_type == "Quantile":
        #     quantiles = np.percentile(X, np.linspace(0, 100, self.nbins + 1), axis=0)
        #     X_quant = np.digitize(X, quantiles, right=False)
        else:
            return X  
        return X_quant


    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        y_train = 2 * y_train - 1
        X_train = self.quantize(X_train)
        if self.eval_set is not None:
            X_val, y_val = self.eval_set
        if X_val is not None:
            X_val = self.quantize(X_val)
            valid_predictions = np.zeros(y_val.shape[0])
            y_val = 2 * y_val - 1
        train_predictions = np.zeros(y_train.shape[0])

        best_val_loss = np.inf
        nothing_improved_rounds = 0
        self.selected_features_list = [] 
        for _ in range(self.n_estimators):
            y_train_loss = 2 * y_train - 1
            if y_val is not None:
                y_val_loss = 2 * y_val - 1
            residuals_train = self.loss_derivative(y_train_loss, train_predictions)

            n = X_train.shape[1]
            if self.rsm <= 1.0:
                k = int(self.rsm * n) 
            else:
                k = int(self.rsm)
            selected_features = np.random.choice(n, k, replace=False)
            self.selected_features_list.append(selected_features)
            X_train_sel = X_train[:, selected_features]

            new_prediction = self.partial_fit(X_train_sel, y_train, residuals_train)
            gamma = self.find_optimal_gamma(residuals_train, train_predictions, new_prediction)
            self.gammas.append(gamma)
            train_predictions += self.learning_rate * gamma * new_prediction
            if X_val is not None:
                valid_predictions += self.models[-1].predict(X_val[:, selected_features]) * self.learning_rate * gamma
                val_loss = self.loss_fn(y_val_loss, valid_predictions)
                self.history['val_loss'].append(val_loss)
            
            train_loss = self.loss_fn(y_train_loss, train_predictions)
            self.history['train_loss'].append(train_loss)
        
            if self.early_stopping_rounds is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    nothing_improved_rounds = 0
                else:
                    nothing_improved_rounds += 1
                if nothing_improved_rounds >= self.early_stopping_rounds:
                    break

        if plot:
            self.plot_history(...)          


    def predict_proba(self, X):
        if self.quantization_type is not None:
            X = self.quantize(X)
        train_predictions = np.zeros(X.shape[0]) # нулевая начальная модель 
    
        for model, gamma, selected_features in zip(self.models, self.gammas, self.selected_features_list):
            train_predictions += self.learning_rate * gamma * model.predict(X[:, selected_features])
    
        proba = self.sigmoid(train_predictions)
        return np.column_stack([1 - proba, proba])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X=None, y=None):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        if 'val_loss' in self.history:
            plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
