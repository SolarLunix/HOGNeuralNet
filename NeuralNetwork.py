import numpy as np

class NN:
    def __init__(self, tf, tl, hl=147, hlt='tanh', olt='ReLU', lr=0.01, it=9000):
        np.random.seed(13)

        self.cache = {
            "A0": np.array(tf).T,
            "Y": np.array(tl).T
                      }

        self.hlt = hlt  # Hidden Layer activation Type
        self.olt = olt  # Output Layer activation Type

        self.n = [self.cache["A0"].shape[0], hl, self.cache["Y"].shape[0]]
        self.m = self.cache["A0"].shape[1]

        print(self.n, self.m)
        self.lr = lr    # Learning Rate
        self.it = it    # Iterations

        self.W1 = np.random.randn(self.n[1], self.n[0]) * 0.001
        self.b1 = np.zeros((self.n[1], 1))
        self.W2 = np.random.randn(self.n[2], self.n[1]) * 0.001
        self.b2 = np.zeros((self.n[2], 1))

    def train(self):
        for i in range(self.it):
            self.forward()
            c = self.cost()
            self.backward()

            if i == 0:
                self.cache["sCost"] = c
            elif i == self.it - 1:
                self.cache["eCost"] = c
            elif i % 1000 == 0:
                print("Cost:", c)

        print("Start Cost:", self.cache["sCost"])
        print("Final Cost:", self.cache["eCost"])

    def forward(self):
        Z1 = np.dot(self.W1, self.cache['A0']) + self.b1
        A1 = self.activation(Z1, self.hlt)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.activation(Z2, self.olt)

        self.cache["Z1"] = Z1
        self.cache["A1"] = A1
        self.cache["Z2"] = Z2
        self.cache["A2"] = A2

    def activation(self, Z, atype):
        A = np.zeros_like(Z)
        if atype == "lReLU":
            A = np.maximum(0.01*Z, Z)
        elif atype == "ReLU":
            A = np.maximum(0, Z)
        elif atype == "sig":
            A = 1 / (1 + np.exp(-Z))
        elif atype == "tanh":
            A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))
        return A

    def cost(self):
        log_probs = (np.multiply(self.cache["Y"], np.log(self.cache["A2"]))) + (np.multiply((1-self.cache["Y"]),np.log(1-self.cache["A2"])))
        cost = -(1/self.m) * np.sum(log_probs)
        cost = np.squeeze(cost)
        return cost

    def backward(self):
        dZ2 = self.dervZ2(self.olt)
        dW2 = (1/self.m) * np.dot(dZ2, self.cache["A1"].T)
        db2 = (1/self.m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.dervZ1(np.dot(self.W2.T, dZ2), self.hlt)
        dW1 = (1/self.m) * np.dot(dZ1, self.cache["A0"].T)
        db1 = (1/self.m) * np.sum(dZ1, axis=1, keepdims=True)

        self.update_params(dW1, db1, dW2, db2)

    def dervZ2(self, atype):
        dZ = np.zeros_like(self.cache["A2"])
        if atype == "lReLU":
            dZ = 1 * np.array((self.cache["A2"] > 0))
        elif atype == "ReLU":
            dZ = 1 * np.array((self.cache["A2"] > 0))
        elif atype == "sig":
            dZ = np.dot(self.cache["A2"], (1 - self.cache["A2"]))
        elif atype == "tanh":
            dZ = 1 - self.cache["A2"]**2
        return dZ

    def dervZ1(self, dZ1, atype):
        dZ = dZ1
        if atype == "lReLU":
            dZ = dZ * (1 * np.array((self.cache["A1"] > 0)))
        elif atype == "ReLU":
            dZ = dZ * (1 * np.array((self.cache["A1"] > 0)))
        elif atype == "sig":
            dZ = dZ * (np.dot(self.cache["A1"], (1 - self.cache["A1"])))
        elif atype == "tanh":
            dZ = dZ * (1 - self.cache["A1"] ** 2)
        return dZ

    def update_params(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.lr * dW1
        self.b1 = self.b1 - self.lr * db1
        self.W2 = self.W2 - self.lr * dW2
        self.b2 = self.b2 - self.lr * db2

    def predict(self, tf):
        self.cache["A0"] = np.array(tf).T
        self.forward()

    def accuracy(self, tl):
        tl = np.array(tl).T
        acc = float((np.dot(tl, self.cache["A2"].T) + np.dot(1-tl, 1-self.cache["A2"].T))/float(tl.size)*100)
        return acc
