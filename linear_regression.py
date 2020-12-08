'''Linear regression: find the relationship between the dependant variable (y) and the independant variable (x)'''

# f(x) = 2x + 5
training_set = [[1, 7], [5, 15], [11, 27], [20, 45], [29, 63], [34, 73], [1000, 2005]]

# Parameters that need to be optimized (weight and bias)
a = 0
b = 0

# Value used for partial derivatives
h = 0.01

def sign(x):
    '''Get sign of number'''
    return (x > 0) - (x < 0)

def f(x, a, b):
    '''Linear function'''
    return a*x + b

def mse(a, b):
    '''Mean Squared Error function, calculates error between known values and predicted values'''
    n = len(training_set)
    sum = 0
    for p in training_set:
        sum += pow(p[1] - f(p[0], a, b), 2)
    return sum/n

def gradient(epochs, lr):
    '''Gradient descent function, used to find the local minimum of a function (in this case, the MSE function)'''
    global a, b
    for i in range(epochs):
        da = (mse(a + h, b) - mse(a, b))/h
        db = (mse(a, b + h) - mse(a, b))/h
        a -= sign(da)*lr
        b -= sign(db)*lr
        print("Epoch", i + 1, "of", epochs, "| a:", a, "b:", b)

# Train
gradient(500, 0.01)