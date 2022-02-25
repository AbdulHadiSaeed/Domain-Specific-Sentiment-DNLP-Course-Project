# EXPERIMENT: PREDICTION OF NON-ENGLISH WORDS FROM ALIGNED VECTORS
import pickle
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from keras.models import load_model

from models.hparams import *


# LOAD VAD SCORES FOR ENGLISH, SPANISH, FRENCH AND POLISH
df_VAD_en = pd.read_csv('../data/vader_lexicon.txt', sep='\t')
df_VAD_en.columns = ['Tokens', 'mean_score', 'std', 'raw_score']

df_VAD_es = pd.read_excel('../data/multi-lingual-aligned-vectors/spanish.xlsx')
df_VAD_es.columns = ['Tokens'] + df_VAD_es.columns[1:].tolist()

df_VAD_fr = pd.read_excel('../data/multi-lingual-aligned-vectors/french.xlsx')
df_VAD_fr.columns = ['Tokens'] + df_VAD_fr.columns[1:].tolist()

df_VAD_pl = pd.read_excel('../data/multi-lingual-aligned-vectors/polish.xls')
df_VAD_pl.columns = ['Tokens'] + df_VAD_pl.columns[1:].tolist()


# LOAD ALIGNED-EN EMBEDDINGS
alignedEnPath = '../data/multi-lingual-aligned-vectors/aligned_embeddings_en.pkl'
with open(alignedEnPath, 'rb') as f:
    aligned_en_embeddings = pickle.load(f)

# FILTER VAD TOKENS WITH ALIGNED TOKENS FOR EN
vocab_list_aligned_en = list()
valence_scores_en = list()
for token in df_VAD_en['Tokens']:
    if token in aligned_en_embeddings.keys() and len(aligned_en_embeddings[token]) == 300:
        vocab_list_aligned_en.append(token)
        valence_scores_en.append(float(df_VAD_en.loc[df_VAD_en['Tokens'] == token, 'mean_score'].values[0]))
    else:
        print(token)
sentiment_en_tokens = {'Tokens': vocab_list_aligned_en}
sentiment_en_df = pd.DataFrame(sentiment_en_tokens)


# LOAD ALIGNED-ES EMBEDDINGS
alignedEsPath = '../data/multi-lingual-aligned-vectors/aligned_embeddings_es.pkl'
with open(alignedEsPath, 'rb') as f:
    aligned_es_embeddings = pickle.load(f)

# FILTER VAD TOKENS WITH ALIGNED TOKENS FOR ES
vocab_list_aligned_es = list()
valence_scores_es = list()
for token in df_VAD_es['Tokens']:
    if token in aligned_es_embeddings.keys() and len(aligned_es_embeddings[token]) == 300:
        vocab_list_aligned_es.append(token)
        valence_scores_es.append(float(df_VAD_es.loc[df_VAD_es['Tokens'] == token, 'Val_Mn']))
    else:
        print(token)
sentiment_es_tokens = {'Tokens': vocab_list_aligned_es}
sentiment_es_df = pd.DataFrame(sentiment_es_tokens)


# LOAD ALIGNED-FR EMBEDDINGS
alignedFrPath = '../data/multi-lingual-aligned-vectors/aligned_embeddings_fr.pkl'
with open(alignedFrPath, 'rb') as f:
    aligned_fr_embeddings = pickle.load(f)

# FILTER VAD TOKENS WITH ALIGNED TOKENS FOR FR
vocab_list_aligned_fr = list()
valence_scores_fr = list()
for token in df_VAD_fr['Tokens']:
    if token in aligned_fr_embeddings.keys() and len(aligned_fr_embeddings[token]) == 300:
        vocab_list_aligned_fr.append(token)
        valence_scores_fr.append(float(df_VAD_fr.loc[df_VAD_fr['Tokens'] == token, 'Valence all (Mean)']))
    else:
        print(token)
sentiment_fr_tokens = {'Tokens': vocab_list_aligned_fr}
sentiment_fr_df = pd.DataFrame(sentiment_fr_tokens)


# LOAD ALIGNED-PL EMBEDDINGS
alignedPlPath = '../data/multi-lingual-aligned-vectors/aligned_embeddings_pl.pkl'
with open(alignedPlPath, 'rb') as f:
    aligned_pl_embeddings = pickle.load(f)

# FILTER VAD TOKENS WITH ALIGNED TOKENS FOR PL
vocab_list_aligned_pl = list()
valence_scores_pl = list()
for token in df_VAD_pl['Tokens']:
    if token in aligned_pl_embeddings.keys() and len(aligned_pl_embeddings[token]) == 300:
        vocab_list_aligned_pl.append(token)
        valence_scores_pl.append(float(df_VAD_pl.loc[df_VAD_pl['Tokens'] == token, 'Valence all (Mean)']))
    else:
        print(token)
sentiment_pl_tokens = {'Tokens': vocab_list_aligned_pl}
sentiment_pl_df = pd.DataFrame(sentiment_pl_tokens)


# LOAD AM DATA WITH DOMAIN-SPECIFIC SENTIMENT SCORES [TRAIN DATA]
domain_path = '../data/multi-domain-am/domain_list'

domain_list = []
with open(domain_path, 'r') as f:
    for line in f:
        domain_list.append(line[:-1])
    domain_list[-1] = line
domain_list = domain_list[1:]

# INITIALIZE LISTS TO STORE CORRELATION VALUES
list_corr_en = list()
list_corr_es = list()
list_corr_fr = list()
list_corr_pl = list()

# PREDICT ON EACH DOMAIN SEPARATELY
for domain_index in tqdm(range(len(domain_list))):
    # DEFINE MODEL LOCATION
    model_path = '../pretrained/ALIGNED/Domain_' + domain_list[domain_index] + '_Predictor.h5'
    # LOAD MODEL
    if MODEL_MODE == 'USE_PRE_TRAINED' and os.path.isfile(model_path):
        model = load_model(model_path)
    else:
        raise ValueError('Only pre-trained mode is allowed and also check if the pretrained models are there.')

    # PREDICTIONS FOR EN
    features = [aligned_en_embeddings[token] for token in vocab_list_aligned_en]
    domain_sentiment_scores = np.reshape(model.predict(np.array(features)), (len(vocab_list_aligned_en, )))
    # SAVE PREDICTIONS INTO DATAFRAME
    sentiment_en_df[domain_list[domain_index].replace(' ', '_')] = domain_sentiment_scores.tolist()
    # EVALUATE CORRELATION WITH VAD LEXICON
    list_corr_en.append(np.corrcoef(domain_sentiment_scores, valence_scores_en)[0][1])

    # PREDICTIONS FOR ES
    features = [aligned_es_embeddings[token] for token in vocab_list_aligned_es]
    domain_sentiment_scores = np.reshape(model.predict(np.array(features)), (len(vocab_list_aligned_es, )))
    # SAVE PREDICTIONS INTO DATAFRAME
    sentiment_es_df[domain_list[domain_index].replace(' ', '_')] = domain_sentiment_scores.tolist()
    # EVALUATE CORRELATION WITH VAD LEXICON
    list_corr_es.append(np.corrcoef(domain_sentiment_scores, valence_scores_es)[0][1])

    # PREDICTIONS FOR FR
    features = [aligned_fr_embeddings[token] for token in vocab_list_aligned_fr]
    domain_sentiment_scores = np.reshape(model.predict(np.array(features)), (len(vocab_list_aligned_fr, )))
    # SAVE PREDICTIONS INTO DATAFRAME
    sentiment_fr_df[domain_list[domain_index].replace(' ', '_')] = domain_sentiment_scores.tolist()
    # EVALUATE CORRELATION WITH VAD LEXICON
    list_corr_fr.append(np.corrcoef(domain_sentiment_scores, valence_scores_fr)[0][1])

    # PREDICTIONS FOR PL
    features = [aligned_pl_embeddings[token] for token in vocab_list_aligned_pl]
    domain_sentiment_scores = np.reshape(model.predict(np.array(features)), (len(vocab_list_aligned_pl, )))
    # SAVE PREDICTIONS INTO DATAFRAME
    sentiment_pl_df[domain_list[domain_index].replace(' ', '_')] = domain_sentiment_scores.tolist()
    # EVALUATE CORRELATION WITH VAD LEXICON
    list_corr_pl.append(np.corrcoef(domain_sentiment_scores, valence_scores_pl)[0][1])

# FINAL CORRELATION DATAFRAME
correlation_df = pd.DataFrame({'Domains': domain_list, 'English': list_corr_en, 'Spanish': list_corr_es,
                               'French': list_corr_fr, 'Polish': list_corr_pl})


# SAVE PREDICTIONS
if WRITE_CSV:
    print('\nWriting into CSV')
    correlation_df.to_csv('../outputs/multilingual_correlation.csv')
    sentiment_en_df.to_csv('../outputs/predicted_english_sentiment.csv')
    sentiment_es_df.to_csv('../outputs/predicted_spanish_sentiment.csv')
    sentiment_fr_df.to_csv('../outputs/predicted_french_sentiment.csv')
    sentiment_pl_df.to_csv('../outputs/predicted_polish_sentiment.csv')
if WRITE_TSV:
    print('\nWriting into TSV')
    correlation_df.to_csv('../outputs/multilingual_correlation.tsv', sep='\t')
    sentiment_en_df.to_csv('../outputs/predicted_english_sentiment.tsv', sep='\t')
    sentiment_es_df.to_csv('../outputs/predicted_spanish_sentiment.tsv', sep='\t')
    sentiment_fr_df.to_csv('../outputs/predicted_french_sentiment.tsv', sep='\t')
    sentiment_pl_df.to_csv('../outputs/predicted_polish_sentiment.tsv', sep='\t')


# PLOT THE DATA INTO A BAR-PLOT
if VISUAL:
    dict_modified = {'Domains': domain_list + domain_list + domain_list + domain_list,
                     'Language': ['English' for _ in range(24)] + ['Spanish' for _ in range(24)] +
                                 ['French' for _ in range(24)] + ['Polish' for _ in range(24)],
                     'Correlation': correlation_df['English'].tolist() + correlation_df['Spanish'].tolist() +
                                    correlation_df['French'].tolist() + correlation_df['Polish'].tolist()
                     }
    modified_df = pd.DataFrame(dict_modified)

    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid")
    chart = sns.barplot(x="Domains", y="Correlation", hue="Language", data=modified_df)

    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=20,
        horizontalalignment='right',
        fontweight='light'
        # fontsize='x-small'
    )
    plt.tight_layout()
    plt.show()
