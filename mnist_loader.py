def get_mnist(dir):
    '''Return (training_data, test_data) tuple from MNIST data in given directory.'''
    print("Loading MNIST data...")
    # Read training data
    training_data = list(zip(read_images(dir + "/train-images.idx3-ubyte", 60000), read_labels(dir + "/train-labels.idx1-ubyte")))
    # Read test data
    test_data = list(zip(read_images(dir + "/t10k-images.idx3-ubyte", 10000), read_labels(dir + "/t10k-labels.idx1-ubyte")))
    # Return tuple
    print("Data loaded successfully!")
    return training_data, test_data

def read_images(filename, n):
    '''Read n 28*28 images from file.'''
    images = []
    with open(filename, "rb") as f:
        byte = f.read(16)
        byte = f.read(1)
        for i in range(n):
            image = []
            for j in range(784):
                image.append(int.from_bytes(byte, "big")/255)
                byte = f.read(1)
            images.append(image)
    return images

def read_labels(filename):
    '''Read all labels from file.'''
    labels = []
    with open(filename, "rb") as f:
        byte = f.read(8)
        byte = f.read(1)
        while byte:
            label = [0 for x in range(10)]
            label[int.from_bytes(byte, "big")] = 1
            labels.append(label)
            byte = f.read(1)
    return labels