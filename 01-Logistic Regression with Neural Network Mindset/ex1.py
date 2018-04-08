import numpy as np
import matplotlib.pyplot as plt
import h5py

def sigmoid(z):
    g =  1/(1+np.exp(-z))
    return g

def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b

def propagate(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {"dw": dw, "db": db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    assert (Y_prediction.shape == (1, m))
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(dim=X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d

if __name__ == "__main__":

    train_data = h5py.File('train_catvnoncat.h5', "r")
    X_train = np.array(train_data["train_set_x"][:])
    y_train = np.array(train_data["train_set_y"][:])
    y_train = y_train.reshape((1, y_train.shape[0]))

    test_data = h5py.File('test_catvnoncat.h5', "r")
    X_test = np.array(test_data["test_set_x"][:])
    y_test = np.array(test_data["test_set_y"][:])
    y_test = y_test.reshape((1, y_test.shape[0]))

    classes = np.array(test_data["list_classes"][:])
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]
    num_px = X_train.shape[1]

    X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
    X_test_flatten = X_test.reshape(X_test.shape[0], -1).T
    X_train = X_train_flatten/255
    X_test = X_test_flatten/255
    d = model(X_train, y_train, X_test, y_test, num_iterations=2000, learning_rate=0.005, print_cost=True)

    costs = np.squeeze(d['costs'])
    plt.figure(1)
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()

    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    for i in learning_rates:
        print("learning rate is: " + str(i))
        models[str(i)] = model(X_train, y_train, X_test, y_test, num_iterations=1500, learning_rate=i, print_cost=False)
        print('\n' + "-------------------------------------------------------" + '\n')

    plt.figure(2)
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.show()