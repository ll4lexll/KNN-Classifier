import os
import sys
import argparse
import time
import itertools
import numpy as np
import pandas as pd


def euclidean_distance(row1, row2, p):
    result = (np.abs(np.array(row1) - np.array(row2)) ** p).sum() ** (1 / p)
    return result


def most_common(lst):
    return max(set(lst), key=lst.count)


class KnnClassifier:
    train_set_x = []
    train_set_y = []

    def __init__(self, k: int, p: float):
        """
        Constructor for the KnnClassifier.

        :param k: Number of nearest neighbors to use.
        :param p: p parameter for Minkowski distance calculation.
        """
        self.k = k
        self.p = p

        # TODO - Place your student IDs here. Single submitters please use a tuple like so: self.ids = (123456789,)
        self.ids = (320845274, 320845274)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        This method trains a k-NN classifier on a given training set X with label set y.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :param y: A 1-dimensional numpy array of m rows. it is guaranteed to match X's rows in length (|m_x| == |m_y|).
            Array datatype is guaranteed to be np.uint8.
        """

        # TODO - your code here
        self.train_set_x = X
        self.train_set_y = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This method predicts the y labels of a given dataset X, based on a previous training of the model.
        It is mandatory to call KnnClassifier.fit before calling this method.

        :param X: A 2-dimensional numpy array of m rows and d columns. It is guaranteed that m >= 1 and d >= 1.
            Array datatype is guaranteed to be np.float32.
        :return: A 1-dimensional numpy array of m rows. Should be of datatype np.uint8.
        """

        # TODO - your code here
        predicts = []
        for point in X:
            closest_neighbors = []
            for i in range(len(self.train_set_x)):
                dist = euclidean_distance(point, self.train_set_x[i], self.p)
                closest_neighbors.append([dist, self.train_set_y[i]])
            closest_neighbors.sort(key=lambda a_entry: a_entry[1])
            closest_neighbors.sort(key=lambda a_entry: a_entry[0])
            # predicts.append(most_common([row[1] for row in closest_neighbors[:10]]))
            for j in range(self.k, 0, -1):
                try:
                    predicts.append(most_common([row[1] for row in closest_neighbors[:j]]))
                    break
                except:
                    continue
        print(predicts)
        return np.array(predicts)



# def main():
#     print("*" * 20)
#     print("Started HW1_ID1_ID2.py")
#     # Parsing script arguments
#     parser = argparse.ArgumentParser()
#     parser.add_argument('csv', type=str, help='Input csv file path')
#     parser.add_argument('k', type=int, help='k parameter')
#     parser.add_argument('p', type=float, help='p parameter')
#     args = parser.parse_args()
#
#     print("Processed input arguments:")
#     print(f"csv = {args.csv}, k = {args.k}, p = {args.p}")
#
#     print("Initiating KnnClassifier")
#     model = KnnClassifier(k=args.k, p=args.p)
#     print(f"Student IDs: {model.ids}")
#     print(f"Loading data from {args.csv}...")
#     data = pd.read_csv(args.csv, header=None)
#     print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
#     X = data[data.columns[:-1]].values.astype(np.float32)
#     y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)
#
#     print("Fitting...")
#     model.fit(X, y)
#     print("Done")
#     print("Predicting...")
#     y_pred = model.predict(X)
#     print("Done")
#     accuracy = np.sum(y_pred == y) / len(y)
#     print(f"Train accuracy: {accuracy * 100 :.2f}%")
#     print("*" * 20)
def main():
    print("Initiating KnnClassifier")
    model = KnnClassifier(k=5, p=2)
    print(f"Student IDs: {model.ids}")
    print(f"Loading data from {'iris.csv'}...")
    data = pd.read_csv('iris.csv', header=None)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
    X = data[data.columns[:-1]].values.astype(np.float32)
    y = pd.factorize(data[data.columns[-1]])[0].astype(np.uint8)

    print("Fitting...")
    model.fit(X, y)
    print("Done")
    print("Predicting...")
    y_pred = model.predict(X)
    print("Done")
    accuracy = np.sum(y_pred == y) / len(y)
    print(f"Train accuracy: {accuracy * 100 :.2f}%")
    print("*" * 20)

if __name__ == "__main__":
    main()
