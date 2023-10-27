import numpy as np
import pandas as pd

"""
import os
from os.path import abspath, dirname                                        #for some reason python executed scripts in /home/user folder,that is
os.chdir(dirname(abspath("/home/egeds/Desktop/Okul/engr421/0076215.py")))   #why I added this segment, so it can execute the script where this .py file is located.
print(os.path.abspath("."))                                                 #uncomment this and change accordingly if you are having the same problem with your linux machine.
"""

X = np.genfromtxt("hw01_data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("hw01_class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 44727 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    X_train = X[:50000]     #splitting data into two
    y_train = y[:50000]
    X_test = X[50000:]
    y_test = y[50000:]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)




# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    unique_classes, class_counts = np.unique(y, return_counts=True) #we are storing class counts by class in an array to get p. We know unique classes but i wanted to make it modular.
    total_samples = len(y)
    class_priors = class_counts / total_samples
    #print(unique_classes, class_counts, total_samples)
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    nucleotid_type = X.shape[1]
    #print(nucleotid_type, num_classes, unique_classes)

    #initialize arrays
    pAcd = np.zeros((num_classes, nucleotid_type))
    pCcd = np.zeros((num_classes, nucleotid_type))
    pGcd = np.zeros((num_classes, nucleotid_type))
    pTcd = np.zeros((num_classes, nucleotid_type))

    for c in unique_classes:
        #getting the nucleotids in the wanted class c
        X_c = X[y == c]
        for nucleotid in range(nucleotid_type):
            #count the occurrences of each nucleotide for the kind of nucleotide
            counts = np.array([np.sum(X_c[:, nucleotid] == 'A'),
                               np.sum(X_c[:, nucleotid] == 'C'),
                               np.sum(X_c[:, nucleotid] == 'G'),
                               np.sum(X_c[:, nucleotid] == 'T')])

            #calculate the probabilities for each nucleotide
            total_samples_in_class = len(X_c)
            pAcd[c - 1, nucleotid] = counts[0] / total_samples_in_class
            pCcd[c - 1, nucleotid] = counts[1] / total_samples_in_class
            pGcd[c - 1, nucleotid] = counts[2] / total_samples_in_class
            pTcd[c - 1, nucleotid] = counts[3] / total_samples_in_class
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    num_samples = X.shape[0]
    num_classes, nucleotid_type = pAcd.shape

    #initialize an array to store the result
    score_values = np.zeros((num_samples, num_classes))

    for i in range(num_samples):
        for c in range(num_classes):
            score = np.log(class_priors[c])  #initialize with the class prior
            for j in range(nucleotid_type):
                nucleotide = X[i, j]
                if nucleotide == 'A':               #contunuing the formula, nucleotid xi | class c
                    score += np.log(pAcd[c, j])
                elif nucleotide == 'C':
                    score += np.log(pCcd[c, j])
                elif nucleotide == 'G':
                    score += np.log(pGcd[c, j])
                elif nucleotide == 'T':
                    score += np.log(pTcd[c, j])
            #print(score)
            score_values[i, c] = score
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    num_classes = scores.shape[1]
    nucleotid_num = len(y_truth) 
    #print(num_classes, nucleotid_num)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(nucleotid_num):
        true_class = y_truth[i] - 1
        predicted_class = np.argmax(scores[i])

        confusion_matrix[predicted_class, true_class] += 1 #building the matrix
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
