import numpy as np
import matplotlib.pyplot as plt

def create_classification_data(n_features=2, n_labels=2, n_data_label=10): 
    X = np.array([])
    y = np.array([])

    for label in range(n_labels):
        mean = np.random.randint(-100, 100, n_features) 
        cov = np.diag(np.random.randint(0, 10, n_features)) # only autocorrelation for now

        X_sample = np.random.multivariate_normal(mean, cov, size=n_data_label)
        X = np.vstack((X, X_sample)) if X.size else X_sample
    
        y_sample = np.full(n_data_label, label, dtype=int) 
        y = np.vstack((y, y_sample)) if y.size else y_sample

    return X, y


def create_regression_data(n_features=2, noise=0.5, n_samples=10):
    X = np.array([])
    coeficient = np.zeros((n_features,1))

    for feature in range(n_features):
        coeficient[feature,0] = np.random.randint(1,10)   

        X_sample = np.random.rand(n_samples)
        X = np.vstack((X,X_sample)) if X.size else X_sample
    
    noise = np.array([y_noise*noise for y_noise in np.random.randn(n_samples)])
    y = np.dot(X.T, coeficient).T[0] + noise

    return X.T, y, coeficient


def split_data(X, y, ratio=0.2):
    indices = np.arange(len(y))
    np.random.shuffle(indices)

    X = X[indices].reshape(X.shape)
    X_train, X_test = X[:int(len(y) * (1 - ratio))], X[-int(len(y) * ratio):]

    y = y[indices].reshape(y.shape)
    y_train, y_test = y[:int(len(y) * (1 - ratio))], y[-int(len(y) * ratio):]

    return X_train, X_test, y_train, y_test


def plot_regression(X, y, model):
    y_predict = model.predict(X)
    
    plt.title(f"{model.__class__.__name__} model")
    plt.ylabel("y")
    plt.xlabel("X_1") if X.ndim != 1 else plt.xlabel("X")

    x_plot = [x_sample[0] for x_sample in X] if X.ndim != 1 else X
    plt.scatter(x_plot, y, color="k" ,label="Data")
    plt.scatter(x_plot, y_predict, color="red", label="Prediction")

    plt.legend()
    plt.show()


def plot_classification():
    pass