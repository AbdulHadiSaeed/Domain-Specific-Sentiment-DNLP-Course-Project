# EXPERIMENT: COMPUTE DOMAIN-SPECIFIC SENTIMENT CORRELATIONS BETWEEN VADER & AM AND BETWEEN VADER & GLOVE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.hparams import *

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

# LOAD VADER LEXICON
vaderPath = '../data/vader_lexicon.txt'
columns_names = ['token', 'mean_score', 'std', 'raw_score']
vader_lexi = pd.read_csv(vaderPath, sep='\t', names=columns_names)
vader_tokens = vader_lexi['token']
vader_senti = vader_lexi['mean_score']
vader_tokens_list = vader_tokens.values

# Load AM SENTIMENT SCORES
am_path = '../data/multi-domain-am/embedding_am.txt'
columns_names = ['Tokens']+domain_list
am_lexi = pd.read_csv(am_path, sep=' ', names=columns_names)

# Load GLOVE SENTIMENT SCORES
glove_path = '../outputs/predicted_sentiments_glove.csv'
glove_lexi = pd.read_csv(glove_path)

# CALCULATE CORRELATION BETWEEN VADER & AM AND BETWEEN VADER & GLOVE
vader_am_domain_corr = list()
vader_glove_domain_corr = list()

# APPEND VADER VOCABULARIES AT THE END OF AM AND GLOVE
vader_padding_dict = {'Tokens': vader_tokens_list}
for domain in domain_list:
    vader_padding_dict[domain] = len(vader_tokens_list) * [0]
vader_padding = pd.DataFrame(vader_padding_dict)
am_lexi = pd.concat([am_lexi, vader_padding], ignore_index=True)
glove_lexi = pd.concat([glove_lexi, vader_padding], ignore_index=True)

# FILTER AM VOCABULARIES WITH VADER VOCABULARIES
refined_am = am_lexi.loc[am_lexi['Tokens'].isin(vader_tokens_list)]. \
    drop_duplicates(subset='Tokens', keep='first', inplace=False). \
    sort_values(by=['Tokens'])

# FILTER GLOVE VOCABULARIES WITH VADER VOCABULARIES
refined_glove = glove_lexi.loc[glove_lexi['Tokens'].isin(vader_tokens_list)]. \
    drop_duplicates(subset='Tokens', keep='first', inplace=False). \
    sort_values(by=['Tokens'])

# FILTER VADER WITH REFINED AM VOCABULARIES; BASICALLY TO GET RID OF DUPLICATES IN VADER
refined_am_tokens = refined_am['Tokens']
refined_am_tokens_list = refined_am_tokens.values
am_refined_vader = vader_lexi.loc[vader_lexi['token'].isin(refined_am_tokens_list)]. \
    drop_duplicates(subset='token', keep='first', inplace=False). \
    sort_values(by=['token'])

# FILTER VADER WITH REFINED GLOVE VOCABULARIES; BASICALLY TO GET RID OF DUPLICATES IN VADER
refined_glove_tokens = refined_glove['Tokens']
refined_glove_tokens_list = refined_glove_tokens.values
glove_refined_vader = vader_lexi.loc[vader_lexi['token'].isin(refined_glove_tokens_list)]. \
    drop_duplicates(subset='token', keep='first', inplace=False). \
    sort_values(by=['token'])

# COMPUTE DOMAIN-WISE CORRELATIONS
for domain in domain_list:
    am_vader_senti_list = am_refined_vader['mean_score'].values
    am_senti_list = refined_am[domain].values
    glove_vader_senti_list = glove_refined_vader['mean_score'].values
    glove_senti_list = refined_glove[domain].values
    vader_am_domain_corr.append(np.corrcoef(am_vader_senti_list, am_senti_list)[0][1])
    vader_glove_domain_corr.append(np.corrcoef(glove_vader_senti_list, glove_senti_list)[0][1])

# SAVE INTO CSV
if WRITE_CSV:
    print('\nWriting into CSV')
    senti_correlation = dict()
    senti_correlation['Domains'] = domain_list
    senti_correlation['VADER_AM'] = vader_am_domain_corr
    senti_correlation['VADER_GLOVE'] = vader_glove_domain_corr
    senti_corr_df = pd.DataFrame(senti_correlation).set_index('Domains')
    senti_corr_df.to_csv("../outputs/corr_vader_am_glove.csv")

# PLOTs
if VISUAL:
    dict_modified = {'Domains': domain_list + domain_list,
                     'Lexicon': ['Seed data' for _ in range(24)] + ['Predicted Lexicon with GloVe' for _ in range(24)],
                     'Correlation with VADER': vader_am_domain_corr + vader_glove_domain_corr
                     }
    modified_df = pd.DataFrame(dict_modified)

    plt.figure(figsize=(10, 5))
    sns.set(style="whitegrid")
    chart = sns.barplot(x="Domains", y="Correlation with VADER", hue="Lexicon", data=modified_df)

    chart.set_xticklabels(
        chart.get_xticklabels(),
        rotation=20,
        horizontalalignment='right',
        fontweight='light'
        # fontsize='x-small'
    )
    plt.tight_layout()
    plt.show()
