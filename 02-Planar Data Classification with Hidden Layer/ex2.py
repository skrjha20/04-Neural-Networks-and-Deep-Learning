import numpy as np
from sklearn.datasets import make_circles,make_blobs, make_moons, make_gaussian_quantiles
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

def sigmoid(z):
    g =  1./(1+np.exp(-z))
    return g

def load_planar_dataset():
    np.random.seed(1)
    m = 400
    N = int(m/2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y

def load_extra_datasets():
    N = 200
    noisy_circles = make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = make_moons(n_samples=N, noise=.2)
    blobs = make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

def logistic_regression(X,Y):
    model = LogisticRegressionCV()
    y_pred = model.fit(X.T, Y.ravel()).predict(X.T)
    acc_score = accuracy_score(Y.ravel(), y_pred)
    print('Accuracy of logistic regression: %d ' % float(acc_score * 100))
    plt.figure(2)
    plot_decision_boundary(lambda x: model.predict(x), X, Y.ravel())
    plt.title("Logistic Regression")
    plt.show()

def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]  # size of output layer
    return (n_x, n_h, n_y)

def layer_sizes_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(5, 3)
    Y_assess = np.random.randn(2, 3)
    return X_assess, Y_assess

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def initialize_parameters_test_case():
    n_x, n_h, n_y = 2, 4, 1
    return n_x, n_h, n_y

def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
        [-0.02136196,  0.01640271],
        [-0.01793436, -0.00841747],
        [ 0.00502881, -0.01245288]]),
     'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
     'b1': np.array([[ 0.], [ 0.], [ 0.], [ 0.]]),
     'b2': np.array([[ 0.]])}
    return X_assess, parameters

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -1 / m * np.sum(logprobs)
    cost = np.squeeze(cost)
    assert (isinstance(cost, float))
    return cost

def compute_cost_test_case():
    np.random.seed(1)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                  [-0.02136196, 0.01640271],
                                  [-0.01793436, -0.00841747],
                                  [0.00502881, -0.01245288]]),
                  'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
                  'b1': np.array([[0.], [0.], [0.], [0.]]),
                  'b2': np.array([[0.]])}
    a2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))
    return a2, Y_assess, parameters

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    dZ2 = A2 - Y.ravel()
    dW2 = (1/m)*np.dot(dZ2, A1.T)
    db2 = (1/m)*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.square(A1)))
    dW1 = (1/m)*np.dot(dZ1, X.T)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                                    [-0.02136196,  0.01640271],
                                    [-0.01793436, -0.00841747],
                                    [ 0.00502881, -0.01245288]]),
                 'W2': np.array([[-0.01057952, -0.00909008,  0.00551454,  0.02292208]]),
                 'b1': np.array([[ 0.],[ 0.],[ 0.],[ 0.]]),
                 'b2': np.array([[ 0.]])}
    cache = {'A1': np.array([[-0.00616578,  0.0020626 ,  0.00349619],
                                [-0.05225116,  0.02725659, -0.02646251],
                                [-0.02009721,  0.0036869 ,  0.02883756],
                                [ 0.02152675, -0.01385234,  0.02599885]]),
            'A2': np.array([[ 0.5002307 ,  0.49985831,  0.50023963]]),
            'Z1': np.array([[-0.00616586,  0.0020626 ,  0.0034962 ],
                            [-0.05229879,  0.02726335, -0.02646869],
                            [-0.02009991,  0.00368692,  0.02884556],
                            [ 0.02153007, -0.01385322,  0.02600471]]),
            'Z2': np.array([[ 0.00092281, -0.00056678,  0.00095853]])}
    return parameters, cache, X_assess, Y_assess

def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return parameters

def update_parameters_test_case():
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
                                    [-0.02311792,  0.03137121],
                                    [-0.0169217 , -0.01752545],
                                    [ 0.00935436, -0.05018221]]),
                    'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
                    'b1': np.array([[ -8.97523455e-07], [  8.15562092e-06], [  6.04810633e-07], [ -2.54560700e-06]]),
                    'b2': np.array([[  9.14954378e-05]])}
    grads = {'dW1': np.array([[ 0.00023322, -0.00205423],
                                [ 0.00082222, -0.00700776],
                                [-0.00031831,  0.0028636 ],
                                [-0.00092857,  0.00809933]]),
                'dW2': np.array([[ -1.75740039e-05,   3.70231337e-03,  -1.25683095e-03, -2.55715317e-03]]),
                'db1': np.array([[  1.05570087e-07], [ -3.81814487e-06], [ -1.90155145e-07], [  5.46467802e-07]]),
                'db2': np.array([[ -1.08923140e-05]])}
    return parameters, grads

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
    return parameters

def nn_model_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.random.randn(1, 3)
    return X_assess, Y_assess

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)
    return predictions

def predict_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    parameters = {'W1': np.array([[-0.00615039,  0.0169021 ],
        [-0.02311792,  0.03137121],
        [-0.0169217 , -0.01752545],
        [ 0.00935436, -0.05018221]]),
     'W2': np.array([[-0.0104319 , -0.04019007,  0.01607211,  0.04440255]]),
     'b1': np.array([[ -8.97523455e-07],
        [  8.15562092e-06],
        [  6.04810633e-07],
        [ -2.54560700e-06]]),
     'b2': np.array([[  9.14954378e-05]])}
    return parameters, X_assess

if __name__ == "__main__":
    np.random.seed(1)
    X, Y = load_planar_dataset()
    plt.figure(1)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()

    shape_X = X.shape
    shape_Y = Y.shape
    m = X.shape[1]
    print('The shape of X is: ' + str(shape_X))
    print('The shape of Y is: ' + str(shape_Y))
    print('I have m = %d training examples!' % (m))

    logistic_regression(X,Y)
    X_assess, Y_assess = layer_sizes_test_case()
    (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
    print("The size of the input layer is: n_x = " + str(n_x))
    print("The size of the hidden layer is: n_h = " + str(n_h))
    print("The size of the output layer is: n_y = " + str(n_y))

    n_x, n_h, n_y = initialize_parameters_test_case()
    parameters = initialize_parameters(n_x, n_h, n_y)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    X_assess, parameters = forward_propagation_test_case()
    A2, cache = forward_propagation(X_assess, parameters)
    print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))

    A2, Y_assess, parameters = compute_cost_test_case()
    print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

    parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
    grads = backward_propagation(parameters, cache, X_assess, Y_assess)
    print("dW1 = " + str(grads["dW1"]))
    print("db1 = " + str(grads["db1"]))
    print("dW2 = " + str(grads["dW2"]))
    print("db2 = " + str(grads["db2"]))

    parameters, grads = update_parameters_test_case()
    parameters = update_parameters(parameters, grads)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    X_assess, Y_assess = nn_model_test_case()
    parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    parameters, X_assess = predict_test_case()
    predictions = predict(parameters, X_assess)
    print("predictions mean = " + str(np.mean(predictions)))

    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    plt.figure(3)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y.ravel())
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()
    predictions = predict(parameters, X)
    acc_score = accuracy_score(Y.ravel(), predictions.ravel())
    print('Accuracy of logistic regression: %d ' % float(acc_score * 100))

    plt.figure(figsize=(16, 32))
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    for i, n_h in enumerate(hidden_layer_sizes):
        parameters = nn_model(X, Y, n_h, num_iterations=5000)
        predictions = predict(parameters, X)
        accuracy = accuracy_score(Y.ravel(), predictions.ravel())
        print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

    #    Performance on other datasets

    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {"noisy_circles": noisy_circles,
                "noisy_moons": noisy_moons,
                "blobs": blobs,
                "gaussian_quantiles": gaussian_quantiles}
    dataset = "noisy_moons"
    X, Y = datasets[dataset]
    X, Y = X.T, Y.reshape(1, Y.shape[0])
    if dataset == "blobs":
        Y = Y % 2
    plt.figure(4)
    plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral)
    plt.show()


