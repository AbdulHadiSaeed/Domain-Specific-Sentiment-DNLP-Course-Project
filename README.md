# Domain-Specific-Sentiment-DNLP-Course-Project

# DomainSpecificLexicons

In this project we implemented the paper "Domain-Specific Sentiment Lexicons Induced from Labeled Documents" (https://aclanthology.org/2020.coling-main.578/) and then performing some additional experiment.




# Data
Several data sources have been used to run experiments.

VADER lexicon. (https://github.com/cjhutto/vaderSentiment) <br/>
Amazon multi-domain sentiment dataset. <br/>
Aligned vectors from fastText. (https://fasttext.cc/docs/en/aligned-vectors.html) <br/>
Word embeddings from fastText. <br/>
Word embeddings from GloVe. <br/>








# To run this you need to have following requirements:

absl-py==0.8.1
astor==0.8.0
certifi==2019.11.28
cycler==0.10.0
gast==0.2.2
google-pasta==0.1.8
grpcio==1.16.1
h5py==2.10.0
joblib==0.14.1
Keras==2.2.4
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
kiwisolver==1.1.0
Markdown==3.1.1
matplotlib==3.1.3
mkl-fft==1.0.15
mkl-random==1.1.0
mkl-service==2.3.0
numpy==1.18.1
opt-einsum==3.1.0
pandas==1.0.0
protobuf==3.11.2
pyparsing==2.4.6
python-dateutil==2.8.1
pytz==2019.3
PyYAML==5.3
scikit-learn==0.22.1
scipy==1.4.1
seaborn==0.10.0
six==1.14.0
tensorboard==1.15.0
tensorflow==1.15.0
tensorflow-estimator==1.15.1
termcolor==1.1.0
tornado==6.0.3
tqdm==4.42.1
webencodings==0.5.1
Werkzeug==0.16.1
wrapt==1.11.2

Setting WRITE_CSV as True will over-ride previously written CSV file. So, keeping a backup of these CSV files might be a good idea. 
Setting LOAD_PRE_TRAINED as False will train the predictor from the start and over-ride the previously saved model. So, keeping a backup of these pretrained models might be a good idea.

Note: The dataset size was big so some dataset file weren't uploaded on github but you can find full code and data set on google drive too.
https://drive.google.com/drive/folders/1hUf6nRSZEUnmR4gzv670w_UJNl1-WmSO?usp=sharing


