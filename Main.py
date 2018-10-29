import time
import Images
import warnings
import NeuralNetwork


def warn(*args, **kwargs):
    pass


t = {"Start": time.time()}
print("Program Starting")
warnings.warn = warn    # Suppress SKLearn deprecation warnings

# Read in and split the images:
imgs = Images.Imgs()

train_fe = imgs.train_fe
train_lb = imgs.train_lb
test_fe = imgs.test_fe
test_lb = imgs.test_lb

t["ReadIn"] = time.time()

print("Running HOG")
train_fe = Images.run_hog(train_fe)
test_fe = Images.run_hog(test_fe)

t["HOG"] = time.time()

print("Training Neural Network")
NN = NeuralNetwork.NN(train_fe, train_lb)
NN.train()

t["Train"] = time.time()

print("Making Predictions")
NN.predict(test_fe)
acc = NN.accuracy(test_lb)
print("\nAccuracy:", round(acc, 2), "%")

t["Test"] = time.time()

t["End"] = time.time()
print("\nTimes:")
print("\t", round(t["ReadIn"] - t["Start"], 5), "\tRead In Time")
print("\t", round(t["HOG"] - t["ReadIn"], 5), "\tHOG Transformation Time")
print("\t", round(t["Train"] - t["HOG"], 5), "\tTraining Time")
print("\t", round(t["Test"] - t["Train"], 5), "\tTraining Time")
print("\t---------------------")
print("\t", round(t["End"] - t["Start"], 5), "\tTotal Time")
