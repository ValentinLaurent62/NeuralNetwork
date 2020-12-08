import ann
import mnist_loader

training_data, test_data = mnist_loader.get_mnist("D:/Dev/Python/AI/data")
net = ann.Network([784, 100, 10])
net.backprop(training_data, 30, 3, test_data)