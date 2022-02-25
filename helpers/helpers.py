import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from models.hparams import *


# FILTER TOKENS OVER DEFINED FREQUENCY
def filter_with_freq(x, y, y_freq, freq_threshold):
    new_x = []
    new_y = []
    for (token, score, freq) in zip(x, y, y_freq):
        if freq >= freq_threshold:
            new_x.append(token)
            new_y.append(score)
        else:
            pass

    return pd.Series(new_x), pd.Series(new_y)


# FILTER TOKENS WITH WITHIN DEFFINED FREQEUNCY LIMIT
def filter_with_freq_ul(x, y, y_freq, freq_threshold_l, freq_threshold_u):
    new_x = []
    new_y = []
    for (token, score, freq) in zip(x, y, y_freq):
        if freq_threshold_l <= freq <= freq_threshold_u:
            new_x.append(token)
            new_y.append(score)
        else:
            pass

    return pd.Series(new_x), pd.Series(new_y)


# FILTER OUT TOKENS THAT ARE NOT PRESENT IN THE EMBEDDING
def filter_with_embedding(x, y, glove_embeddings):
    new_x = []
    new_y = []
    for (token, score) in zip(x, y):
        if token in glove_embeddings:
            new_x.append(token)
            new_y.append(score)
        else:
            pass

    return pd.Series(new_x), pd.Series(new_y)


# FILTER OUT valid_x and valid_y FROM x
def filter_with_validtest(x, y, valid_x, test_x):
    new_x = []
    new_y = []

    for (token, score) in zip(x, y):
        if not (any(valid_x.isin([token]))) and not (any(test_x.isin([token]))):
            new_x.append(token)
            new_y.append(score)
        else:
            pass

    return pd.Series(new_x), pd.Series(new_y)


# SPLITTING DATA SET INTO TRAIN/VALID/TEST
def split_dataset(x, y, portion_test, portion_valid, polarity_threshold):
    y_polarity = np.ones(y.shape, dtype=np.float16)
    y_polarity[np.bitwise_and(y >= -polarity_threshold, y <= polarity_threshold)] = 0
    y_polarity[y < -polarity_threshold] = -1
    train_valid_x, test_x, train_valid_y, test_y = train_test_split(x, y, test_size=portion_test, stratify=y_polarity,
                                                                    random_state=RAND_SEED)

    y_polarity = np.ones(train_valid_y.shape, dtype=np.float16)
    y_polarity[np.bitwise_and(train_valid_y >= -polarity_threshold, train_valid_y <= polarity_threshold)] = 0
    y_polarity[train_valid_y < -polarity_threshold] = -1
    train_x, valid_x, train_y, valid_y = train_test_split(train_valid_x, train_valid_y, test_size=portion_valid,
                                                          stratify=y_polarity, random_state=RAND_SEED)

    return train_x, valid_x, train_y, valid_y, test_x, test_y, train_valid_x, train_valid_y


# EVALUATE PREDICTION ON VADER LEXICON
def evaluate_vader_model(model, x_space, y_space, vader_lexi, glove_embeddings, std_multipliers, difference_range):
    features = [glove_embeddings[token] for token in x_space]
    predictions = model.predict(np.array(features))

    difference = np.abs(predictions[:, 0] - y_space)
    std_list = np.array([vader_lexi.loc[vader_lexi['token'] == token]['std'].values[0] for token in x_space],
                        dtype=np.float32)

    value_accuracy = []

    for multiplier in std_multipliers:
        value_accuracy.append(np.sum(difference.values <= multiplier * std_list) / y_space.shape[0])

    combined_polarity = predictions[:, 0] * y_space
    polarity_accuracy = np.sum(combined_polarity >= 0) / y_space.shape[0]

    diff_hist, diff_hist_edge = np.histogram(difference, bins=difference_range)

    return predictions, diff_hist, diff_hist_edge, value_accuracy, polarity_accuracy


# COMPUTE MDS ON CORRELATION MATRIX
def compute_mds(matrix_corr):
    D = 1-matrix_corr
    # Number of points
    n = len(D)

    # Centering matrix
    H = np.eye(n) - np.ones((n, n)) / n

    # YY^T
    B = -H.dot(D ** 2).dot(H) / 2

    # Diagonalize
    evals, evecs = np.linalg.eigh(B)

    # Sort by eigenvalue in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Compute the coordinates using positive-eigenvalued components only
    w, = np.where(evals > 0)
    L = np.diag(np.sqrt(evals[w]))
    V = evecs[:, w]
    Y = V.dot(L)

    return Y, evals
