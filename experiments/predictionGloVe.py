# EXPERIMENT: PREDICTION ON GLOVE VOCABULARIES
import os.path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import load_model

from helpers.helpers import filter_with_embedding, filter_with_freq, split_dataset, compute_mds
from models.hparams import *
from models.models import sentiment_predictor, SentimentScoreGenerator

# LOAD GLOVE EMBEDDING
glovePath = '../data/glove.840B.300d.pkl'
with open(glovePath, 'rb') as f:
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

# LOAD AM DATA WITH DOMAIN-SPECIFIC SENTIMENT SCORES [TRAIN DATA]
domain_path = '../data/multi-domain-am/domain_list'
am_embedding_path = '../data/multi-domain-am/embedding_am.txt'
am_freq_path = '../data/multi-domain-am/embedding_am_freq_pd.txt'

domain_list = []
with open(domain_path, 'r') as f:
    for line in f:
        domain_list.append(line[:-1])
    domain_list[-1] = line

nTokens = int(domain_list[0][1:7])
nDomains = int(domain_list[0][9:11])
domain_list = domain_list[1:]
columns_names = ['tokens'] + domain_list

am_lexi = pd.read_csv(am_embedding_path, sep=' ', names=columns_names)
am_freq = pd.read_csv(am_freq_path, sep=' ', names=columns_names)

# TRAIN ON EACH DOMAIN SEPARATELY WITH FREQUENCY OVER 500; THEN, PREDICT ON TEST DATA
for domain_index in tqdm(range(24)):
    x = am_lexi['tokens'].values
    y = am_lexi[domain_list[domain_index]].values
    y_freq = am_freq[domain_list[domain_index]].values

    # print('Number of tokens after before filtering %d'%x.shape[0])
    x, y = filter_with_freq(x, y, y_freq, freq_threshold=500)
    # print('Number of tokens after freq filtering %d'%x.shape[0])
    x, y = filter_with_embedding(x, y, glove_embeddings)
    # print('Number of tokens after glove filtering %d'%x.shape[0])
    x, y = shuffle(x, y, random_state=RAND_SEED)

    min_score, max_score = np.min(y), np.max(y)

    portion_test, portion_valid = 0.1, 0.1
    # Absolute polarity threshold
    polarity_threshold = 0.1
    train_x, valid_x, train_y, valid_y, test_x, test_y, _, _ = split_dataset(x, y, portion_test, portion_valid,
                                                                             polarity_threshold)

    # DEFINE MODEL
    model_path = '../pretrained/GLOVE/Domain_' + domain_list[domain_index] + '_Predictor.h5'
    checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min', verbose=1,
                                 save_best_only=True, save_weights_only=False)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', mode='min', verbose=1, factor=0.5, min_delta=0.001,
                                       patience=4)
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
        model_name = domain_list[domain_index] + '_Predictor'
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

    # PREDICTIONS
    features = [glove_embeddings[token] for token in vocab_list_glove]
    domain_sentiment_scores = np.reshape(model.predict(np.array(features)), (len(vocab_list_glove, )))
    # MODIFY DOMAIN NAME AND PLACE INTO DICTIONARY
    sentiment_df[domain_list[domain_index].replace(' ', '_')] = domain_sentiment_scores.tolist()

# SAVE PREDICTIONS
if WRITE_CSV:
    print('\nWriting into CSV')
    sentiment_df.to_csv('../outputs/predicted_sentiments_glove.csv')
if WRITE_TSV:
    print('\nWriting into TSV')
    sentiment_df.to_csv('../outputs/predicted_sentiments_glove.tsv', sep='\t')

# COMPUTE CORRELATION COEFFICIENTS BETWEEN PREDICTED SENTIMENT SCORES FROM DIFFERENT DOMAINS
scores = np.empty((nDomains, len(domain_sentiment_scores)), dtype=np.float32)
for domain_index in range(nDomains):
    scores[domain_index] = sentiment_df[domain_list[domain_index].replace(' ', '_')].tolist()

cor_matrix = np.corrcoef(scores)

cor_matrix_df = pd.DataFrame(index=domain_list)
for domain_index in range(nDomains):
    cor_matrix_df[domain_list[domain_index]] = cor_matrix[domain_index, :]

# COMPUTE MDS FROM THE CORRELATION MATRIX
reduced_space, axis_weights = compute_mds(cor_matrix)

# PLOT GRAPHS
if VISUAL:
    # HEAT-MAP
    sns.set()
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    heatmap = sns.heatmap(cor_matrix_df, annot=False, fmt='0.2f', xticklabels=1, annot_kws={'rotation': 90, 'size': 8})
    plt.title('(a) Correlation heatmap', fontsize=16)
    heatmap.set_xticklabels(
        heatmap.get_xticklabels(),
        rotation=50,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )
    heatmap.set_yticklabels(
        heatmap.get_yticklabels(),
        rotation=50,
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'
    )

    # PLOT MDS REPRESENTATION
    ax = fig.add_subplot(122)
    ax.scatter(reduced_space[:, 0], reduced_space[:, 1])
    plt.title('(b) Sentiment-based cross-domain relations in 2D plane', fontsize=16)
    text_offsets = [
        [-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.25, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03,
         -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03, -0.03],
        [-0.017, -0.017, 0.007, 0.007, 0.007, 0.007, -0.017, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007, -0.017, 0.007,
         0.007, 0.007, -0.017, 0.007, 0.007, 0.007, 0.007, 0.007, 0.007]]
    for i, txt in enumerate(domain_list):
        ax.annotate(txt, (reduced_space[i, 0] + text_offsets[0][i], reduced_space[i, 1] + text_offsets[1][i]))
    plt.tight_layout()

    plt.show()
