import sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


class RandomForest:

    def __init__(self, X, Y):
        self.random_forest = RandomForestClassifier()
        self.random_forest.fit(X, Y)

    def best_parameters(self, X, Y):
        random_grid = {'n_estimators': [200, 400, 600],
                       'max_features': ['auto', 'sqrt'],
                       'max_depth': [5, 10, None],
                       'min_samples_split': [2, 10],
                       'min_samples_leaf': [1, 4]}

        self.random_forest = GridSearchCV(self.random_forest, random_grid)
        self.random_forest.fit(X, Y)
        self.random_forest.fit(X, Y)
        return self.random_forest.best_estimator_, self.random_forest.best_score_

    def score(self, X, Y):
        y_pred = self.random_forest.predict(X)
        return sklearn.metrics.accuracy_score(Y, y_pred)

    def predict_proba(self, X):
        return self.random_forest.predict_proba(X)


class MultilayerPerceptron:

    def __init__(self, X, Y):
        self.mlp = MLPClassifier()
        self.mlp.fit(X, Y)

    def best_parameters(self, X, Y):
        random_grid = {
            'learning_rate': ["constant", "invscaling"],
            'hidden_layer_sizes': [(20,), (20, 20, 20),
                                   (40,), (40, 40, 40),
                                   (60,), (60, 60, 60),
                                   (80,), (80, 80, 80)]
        }
        self.mlp = GridSearchCV(self.mlp, random_grid)
        self.mlp.fit(X, Y)
        return self.mlp.best_estimator_, self.mlp.best_score_

    def score(self, X, Y):
        y_pred = self.mlp.predict(X)
        return sklearn.metrics.accuracy_score(Y, y_pred)

    def predict_proba(self, X):
        return self.mlp.predict_proba(X)
