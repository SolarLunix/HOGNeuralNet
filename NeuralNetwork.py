import numpy as np
from sklearn.metrics import accuracy_score

class NN:
    def __init__(self, hl=147, hlt='ReLU', lr=1, it=900):
        np.random.seed(13)
        n = [0, hl, 0]
        self.cache = {
            "n": n,
            "hlt": hlt,
            "olt": 'sig',
            "lr": lr,
            'it': it
        }

    def train(self, tf, tl):
        self.populate(tf, tl)
        for i in range(self.cache["it"]):
            self.forward()
            c = self.loss()
            self.backward()

            if i % 50 == 0:
                print("Cost at", i, c)
                print("Accuracy:", self.accuracy(self.cache["Y"]))

    def populate(self, tf, tl):
        A0 = np.array(tf).T
        Y = np.array(tl).T

        n = self.cache["n"]
        n[0] = A0.shape[0]
        n[2] = Y.shape[0]
        m = A0.shape[1]

        c = 0.001

        self.cache["A0"] = A0
        self.cache["Y"] = Y
        self.cache["n"] = n
        self.cache["m"] = m
        self.cache["W1"] = c * np.random.randn(n[1], n[0])
        self.cache["b1"] = c * np.random.randn(n[1], 1)
        self.cache["W2"] = c * np.random.randn(n[2], n[1])
        self.cache["b2"] = c * np.random.randn(n[2], 1)


    def forward(self):
        self.cache["Z1"] = np.dot(self.cache["W1"], self.cache["A0"]) + self.cache["b1"]
        self.cache["A1"] = self.activation(self.cache["Z1"], self.cache["hlt"])
        self.cache["Z2"] = np.dot(self.cache["W2"], self.cache["A1"]) + self.cache["b2"]
        self.cache["A2"] = self.activation(self.cache["Z2"], self.cache["olt"])

    def activation(self, Z, atype):
        A = np.zeros_like(Z)
        if atype == "lReLU":
            A = np.maximum(0.01 * Z, Z)
        elif atype == "ReLU":
            A = np.maximum(0, Z)
        elif atype == "sig":
            A = 1 / (1 + np.exp(-Z))
        elif atype == "tanh":
            A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
        elif atype == "hard":
            A[A >= 0.5] = 1
            A[A < 0] = 0
        return A

    def derivative(self, A, atype):
        dZ = np.array(A)
        if atype == "lReLU":
            dZ[dZ > 0] = 1
            dZ[dZ <= 0] = 0
        elif atype == "ReLU":
            dZ[dZ > 0] = 1
            dZ[dZ <= 0] = 0.01
        elif atype == "sig":
            dZ = np.dot(A, (1-A))
        elif atype == "tanh":
            dZ = 1 - np.square(A)
        return dZ

    def loss(self):
        Y = self.cache["Y"]
        A2 = self.cache["A2"]
        m = self.cache["m"]

        loss = np.add(np.multiply(Y, np.log(A2)), np.multiply((1-Y), np.log((1-A2))))
        cost = (-1/m) * np.sum(loss)
        cost = np.squeeze(cost)
        self.cache["cost"] = cost
        return cost

    def backward(self):
        m = self.cache["m"]

        self.cache["dZ2"] = np.subtract(self.cache["A2"], self.cache["Y"])
        self.cache["dW2"] = (1/m) * np.dot(self.cache["dZ2"], self.cache["A1"].T)
        self.cache["db2"] = (1/m) * np.sum(self.cache["dZ2"], axis=1, keepdims=True)
        self.cache["dZ1"] = np.dot(self.cache["W2"].T, self.cache["dZ2"]) * self.derivative(self.cache["A1"], self.cache["hlt"])
        self.cache["dW1"] = (1/m) * np.dot(self.cache["dZ1"], self.cache["A0"].T)
        self.cache["db1"] = (1/m) * np.sum(self.cache["dZ1"], axis=1, keepdims=True)

        self.update()

    def update(self):
        self.cache["W1"] = self.cache["W1"] - (self.cache["lr"] * self.cache["dW1"])
        self.cache["b1"] = self.cache["b1"] - (self.cache["lr"] * self.cache["db1"])
        self.cache["W2"] = self.cache["W2"] - (self.cache["lr"] * self.cache["dW2"])
        self.cache["b2"] = self.cache["b2"] - (self.cache["lr"] * self.cache["db2"])

    def predict(self, tf):
        self.cache["A0"] = np.array(tf).T
        self.forward()
        return self.cache["A2"]

    def accuracy(self, tl):
        if(np.array(tl).shape[0] != self.cache["Y"].shape[0]):
            self.cache["Y"] = np.array(tl).T

        self.finalout()
        acc = 0
        for i in range(self.cache["Y"].T.shape[0]):
            if np.array_equal(self.cache["Y"].T[i], self.cache["out"].T[i]):
                acc = acc + 1
        acc = acc / self.cache["Y"].T.shape[0] * 100
        return acc

    def finalout(self):
        max = np.max(self.cache["A2"], axis=0).reshape((1, -1))
        self.cache["out"] = 1 * np.greater_equal(self.cache["A2"], max)


