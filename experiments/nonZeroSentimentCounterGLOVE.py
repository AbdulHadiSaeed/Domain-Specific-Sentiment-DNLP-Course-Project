# EXPERIMENT: COUNT NUMBER OF TOKENS WITH NON-ZERO AND RELEVANT (OVER SOME THRESHOLD) SENTIMENTS
import numpy as np
import pandas as pd
from models.hparams import VISUAL, COMMAND_LINE, WRITE_CSV

# LOAD DOMAINS
domain_path = '../data/multi-domain-am/domain_list'

domain_list = []
with open(domain_path, 'r') as f:
    for line in f:
        domain_list.append(line[:-1])
    domain_list[-1] = line

nTokens = int(domain_list[0][1:7])
nDomains = int(domain_list[0][9:11])
domain_list = domain_list[1:]
columns_names = ['Tokens']+domain_list

# Load AM SENTIMENT SCORES
am_path = '../data/multi-domain-am/embedding_am.txt'
am_lexi = pd.read_csv(am_path, sep=' ', names=columns_names)
am_tokens = am_lexi['Tokens']
am_tokens_list = am_tokens.values

# Load GLOVE SENTIMENT SCORES
glove_path = '../outputs/predicted_sentiments_glove.csv'
glove_lexi = pd.read_csv(glove_path)
glove_tokens = glove_lexi['Tokens']
glove_tokens_list = glove_tokens.values

# COUNT NON ZERO/RELEVANT SENTIMENTS TOKENS
non_zero_senti_counter_am = list()
non_zero_senti_counter_glove = list()
for domain in domain_list:
    non_zero_senti_counter_am.append(f"{np.sum(np.abs(am_lexi[domain].values)>0.2):,d}")
    non_zero_senti_counter_glove.append(f"{np.sum(np.abs(glove_lexi[domain].values) > 0.2):,d}")
    if COMMAND_LINE:
        print(domain, f"{np.sum(np.abs(am_lexi[domain].values)>0.2):,d}",
              f"{np.sum(np.abs(glove_lexi[domain].values) > 0.2):,d}")

# SAVE INTO CSV
if WRITE_CSV:
    print('\nWriting into CSV')
    non_zero_senti_counter = dict()
    non_zero_senti_counter['Domains'] = domain_list
    non_zero_senti_counter['AM'] = non_zero_senti_counter_am
    non_zero_senti_counter['GLOVE'] = non_zero_senti_counter_glove
    senti_corr_df = pd.DataFrame(non_zero_senti_counter).set_index('Domains')
    senti_corr_df.to_csv("../outputs/non_zero_sentiment_counter_glove.csv")
