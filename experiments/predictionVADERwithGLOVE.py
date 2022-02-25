# EXPERIMENT: PREDICTION ON VADER LEXICON WITH GLOVE EMBEDDINGS
import os.path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

from helpers.helpers import filter_with_embedding, split_dataset, evaluate_vader_model
from models.hparams import *
from models.models import sentiment_predictor, SentimentScoreGenerator

# LOAD GLOVE EMBEDDING
glove_path = '../data/glove.840B.300d.pkl'
with open(glove_path, 'rb') as f:
    glove_embeddings = pickle.load(f)

# GLOVE VOCABULARIES FOR DOMAIN-SPECIFIC PREDICTIONS [TEST DATA]
vocab_list = list()
vocab_list_glove = list()
for index, key in enumerate(glove_embeddings.keys()):
    if len(glove_embeddings[key]) == 300:
        vocab_list_glove.append(key)
    else:
        print(key)
sentiment_tokens = {'Tokens': vocab_list_glove}
sentiment_df = pd.DataFrame(sentiment_tokens)

# LOAD VADER LEXICON [TRAIN & TEST DATA]
vader_path = '../data/vader_lexicon.txt'
columns_names = ['token', 'mean_score', 'std', 'raw_score']
vader_lexi = pd.read_csv(vader_path, sep='\t', names=columns_names)
x = vader_lexi['token']
y = vader_lexi['mean_score']

x, y = filter_with_embedding(x, y, glove_embeddings)
x, y = shuffle(x, y, random_state=RAND_SEED)

min_score = np.min(y)
max_score = np.max(y)

portion_test = 0.2
portion_valid = 0.2
polarity_threshold = 0.25
train_x, valid_x, train_y, valid_y, test_x, test_y, train_valid_x, train_valid_y \
    = split_dataset(x, y, portion_test, portion_valid, polarity_threshold)

# DEFINE MODEL
model_path = '../pretrained/VADER/Vader_Predictor_with_GLOVE.h5'
checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', verbose=1,
                             save_best_only=True, save_weights_only=False)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', mode='min', verbose=1, factor=0.5, min_delta=0.001, patience=4)
earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=20)

callback_list = [checkpoint, reduceLROnPlat, earlyStop]

train_generator = SentimentScoreGenerator(train_x, train_y, glove_embeddings, BATCH_SIZE, is_train=True)
valid_generator = SentimentScoreGenerator(valid_x, valid_y, glove_embeddings, BATCH_SIZE, is_train=False)

# LOAD/COMPILE MODEL
if MODEL_MODE == 'USE_PRE_TRAINED' and os.path.isfile(model_path):
    model = load_model(model_path)

elif MODEL_MODE == 'RESUME_TRAINING' and os.path.isfile(model_path):
    model = load_model(model_path)

    # TRAIN
    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_x)) / float(BATCH_SIZE)),
        validation_data=valid_generator,
        validation_steps=np.ceil(float(len(valid_x)) / float(BATCH_SIZE)),
        epochs=N_EPOCHS,
        verbose=0,
        callbacks=callback_list
    )

elif MODEL_MODE == 'RESET_MODEL' or not os.path.isfile(model_path):
    model_name = 'VADER_Predictor'
    model = sentiment_predictor(model_name, min_score, max_score, input_shape=(300,))
    model.compile(optimizer=Adam(lr=1e-3), loss='mse', metrics=['acc'])

    # TRAIN
    model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(train_x)) / float(BATCH_SIZE)),
        validation_data=valid_generator,
        validation_steps=np.ceil(float(len(valid_x)) / float(BATCH_SIZE)),
        epochs=N_EPOCHS,
        verbose=0,
        callbacks=callback_list
    )

else:
    raise NameError('Un-specified mode for model!')


# PREDICTIONS ON WHOLE GloVe
if WRITE_CSV:
    features = [glove_embeddings[token] for token in vocab_list_glove]
    predictions = np.reshape(model.predict(np.array(features)), (len(vocab_list_glove, )))
    # INTO DICTIONARY
    sentiment_df['Scores'] = predictions.tolist()
    print('\nWriting into CSV')
    sentiment_df.to_csv('../outputs/predicted_sentiments_glove_from_VADER.csv')

# PREDICTIONS
features = [glove_embeddings[token] for token in test_x]
predictions = model.predict(np.array(features))

# EVALUATIONS
difference = np.abs(predictions[:, 0]-test_y)
std_list = np.array([vader_lexi.loc[vader_lexi['token'] == token]['std'].values[0]
                     for token in test_x], dtype=np.float32)

std_multipliers = np.arange(0.25, 4.25, 0.25)
difference_range = np.arange(0, 5, 0.25)

# EVALUATION ON TRAIN SPLIT
train_predictions, train_hist, train_hist_edge, train_value_accuracy, train_polarity_accuracy = \
    evaluate_vader_model(model, train_x, train_y, vader_lexi, glove_embeddings, std_multipliers, difference_range)
# EVALUATION ON TEST SPLIT
test_predictions, test_hist, test_hist_edge, test_value_accuracy, test_polarity_accuracy = \
    evaluate_vader_model(model, test_x, test_y, vader_lexi, glove_embeddings, std_multipliers, difference_range)
# EVALUATION ON VALIDATION SPLIT
valid_predictions, valid_hist, valid_hist_edge, valid_value_accuracy, valid_polarity_accuracy = \
    evaluate_vader_model(model, valid_x, valid_y, vader_lexi, glove_embeddings, std_multipliers, difference_range)

if VISUAL:
    # PLOT STRATIFIED SPLIT OF THE VADER DATA
    fig = plt.figure(figsize=(18, 3))
    plt.subplot(1, 3, 1)
    y.hist(label='Whole')
    train_valid_y.hist(label='Train + Valid')
    train_y.hist(label='Train')
    plt.legend(fontsize=13)
    plt.xlabel('Mean Sentiment Score', fontsize=13)
    plt.ylabel('Number of Tokens', fontsize=13)
    plt.title('(a) Stratified Splits', fontsize=20)

    # PLOT FOR ABSOLUTE ERROR vs PREDICTION CDF
    ax1 = fig.add_subplot(132)
    # ax1.plot(train_hist_edge[:-1], np.cumsum(train_hist/sum(train_hist)), label='train')
    ax1.plot(test_hist_edge[:-1], np.cumsum(test_hist/sum(test_hist)), label='Test Split')
    ax1.plot(valid_hist_edge[:-1], np.cumsum(valid_hist/sum(valid_hist)), label='Validation Split', linestyle='--')
    plt.grid(linestyle='dotted')
    plt.xlabel('Absolute Error = |Predicted Score - Mean Opinion Score|', fontsize=13)
    plt.ylabel('% Correct', fontsize=13)
    plt.title('(b) Model performance w.r.t. \n the absolute error', fontsize=20)
    plt.legend(fontsize=13)

    # PLOT FOR STD MULTIPLIER vs ACCURACY
    ax1 = fig.add_subplot(133)
    # ax1.plot(std_multipliers, train_value_accuracy, label='train')
    ax1.plot(std_multipliers, test_value_accuracy, label='Test Split')
    ax1.plot(std_multipliers, valid_value_accuracy, label='Validation Split', linestyle='--')
    plt.grid(linestyle='dotted')
    plt.xlabel('Standard Deviation Multipliers', fontsize=13)
    plt.ylabel('% Correct', fontsize=13)
    plt.title('(c) Model performance w.r.t. \n the STD of human scores', fontsize=20)
    plt.legend(fontsize=13)
    plt.show()


# PEARSON CORRELATIONS
if COMMAND_LINE:
    train_pearsonCoef = np.corrcoef(train_y.values, train_predictions[:, 0])
    print('Pearson Coefficient for train split: %f' % (train_pearsonCoef[0, 1]))

    test_pearsonCoef = np.corrcoef(test_y.values, test_predictions[:, 0])
    print('Pearson Coefficient for test split: %f' % (test_pearsonCoef[0, 1]))

    valid_pearsonCoef = np.corrcoef(valid_y.values, valid_predictions[:, 0])
    print('Pearson Coefficient for valid split: %f' % (valid_pearsonCoef[0, 1]))
