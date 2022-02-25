# EXPERIMENT: EXTRACT THE TOP 20 POSITIVE AND NEGATIVE SENTIMENT TOKENS FROM EACH DOMAIN
import pickle
import pandas as pd
from tqdm import tqdm

from models.hparams import WRITE_CSV

# LOAD GLOVE EMBEDDING
glovePath = '../data/glove.840B.300d.pkl'
with open(glovePath, 'rb') as f:
    glove_embeddings = pickle.load(f)


# LOAD AM DATA WITH DOMAIN-SPECIFIC SENTIMENT SCORES
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
columns_names = ['Tokens'] + domain_list

am_lexi = pd.read_csv(am_embedding_path, sep=' ', names=columns_names)
am_freq = pd.read_csv(am_freq_path, sep=' ', names=columns_names)

max_min_score_dict = dict()

# EXTRACT EXTREME SENTIMENT TOKENS FOR EACH DOMAIN AND FREQUENCY RANGE
freq_range = [2, 10, 25, 50, 100, 500]

for domain_index in tqdm(range(nDomains)):
    for freq_threshold in freq_range:
        domain_score_df = am_lexi[['Tokens', domain_list[domain_index]]]
        domain_freq_df = am_freq[['Tokens', domain_list[domain_index]]]

        merged_df = domain_score_df.merge(domain_freq_df, left_on='Tokens', right_on='Tokens')
        merged_df = merged_df[merged_df[domain_list[domain_index]+'_y'] > freq_threshold]

        merged_df.sort_values(by=[domain_list[domain_index]+'_x'], inplace=True)
        min_df = merged_df.head(20)

        merged_df.sort_values(by=[domain_list[domain_index]+'_x'], ascending=False, inplace=True)
        max_df = merged_df.head(20)

        col_key = domain_list[domain_index]+'_positive_freq_threshold_'+str(freq_threshold)
        max_min_score_dict[col_key] = max_df['Tokens'].values
        col_key = domain_list[domain_index]+'_positive_score_freq_threshold_'+str(freq_threshold)
        max_min_score_dict[col_key] = max_df[domain_list[domain_index]+'_x'].values

        col_key = domain_list[domain_index]+'_negative_freq_threshold_'+str(freq_threshold)
        max_min_score_dict[col_key] = min_df['Tokens'].values
        col_key = domain_list[domain_index]+'_negative_score_freq_threshold_'+str(freq_threshold)
        max_min_score_dict[col_key] = min_df[domain_list[domain_index]+'_x'].values

# SAVE INTO CSV
if WRITE_CSV:
    print('\nWriting into CSV')
    extreme_sentiment_df = pd.DataFrame(max_min_score_dict)
    extreme_sentiment_df.to_csv("../outputs/extreme_sentiments.csv")
