import models.classifier
import models.linear_regressor
import utils
import models


import matplotlib.pyplot as plt
import numpy as np

X, y, coeficients = utils.create_regression_data(n_samples=100)
X_train, X_test, y_train, y_test = utils.split_data(X, y)

model = models.linear_regressor.RidgeRegressor()
model.fit(X_train, y_train)

utils.plot_regression(X_test, y_test, model)