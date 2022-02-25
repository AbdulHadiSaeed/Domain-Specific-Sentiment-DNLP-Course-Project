import numpy as np
from sklearn.utils import shuffle
from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, Dense
from keras.utils import Sequence
from keras.initializers import Constant

from models.hparams import RAND_SEED


# SENTIMENT SCORE PREDICTOR MODEL
# TODO: TRY OTHER MODELS FOR BETTER PREDICTION
def sentiment_predictor(model_name, min_score, max_score, input_shape):
    input_tensor = Input(shape=input_shape, name='input0')

    x = Dense(500, name='fc0')(input_tensor)
    x = BatchNormalization(name='bn0')(x)
    x = Activation('relu', name='a0')(x)
    x = Dropout(0.2, name='dp0')(x)

    x = Dense(500, name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='a1')(x)
    x = Dropout(0.2, name='dp1')(x)

    x = Dense(100, name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='a2')(x)
    x = Dropout(0.2, name='dp2')(x)

    x = Dense(10, name='fc3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('softmax', name='a3')(x)

    x = Dense(1, kernel_initializer=Constant(np.linspace(1.2*min_score, 1.2*max_score, 10)),
              bias_initializer=Constant(0), activation='linear', name='fc4')(x)

    model = Model(inputs=input_tensor, outputs=x, name=model_name)
    return model


# SENTIMENT SCORE PREDICTOR MODEL
# TODO: TRY OTHER MODELS FOR BETTER PREDICTION
def sentiment_class_predictor(model_name, n_classes, input_shape):
    input_tensor = Input(shape=input_shape, name='input0')

    x = Dense(500, name='fc0')(input_tensor)
    x = BatchNormalization(name='bn0')(x)
    x = Activation('relu', name='a0')(x)
    x = Dropout(0.2, name='dp0')(x)

    x = Dense(500, name='fc1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = Activation('relu', name='a1')(x)
    x = Dropout(0.2, name='dp1')(x)

    x = Dense(100, name='fc2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = Activation('relu', name='a2')(x)
    x = Dropout(0.2, name='dp2')(x)

    x = Dense(n_classes, name='fc3')(x)
    x = BatchNormalization(name='bn3')(x)
    x = Activation('softmax', name='a3')(x)

    model = Model(inputs=input_tensor, outputs=x, name=model_name)
    return model


# BACTH GENERATOR FOR SCORE PREDICTION MODEL
class SentimentScoreGenerator(Sequence):
    def __init__(self, tokens, scores, glove_embeddings, batch_size, is_train=False):
        self.tokens = tokens
        self.scores = scores
        self.batch_size = batch_size
        self.is_train = is_train
        self.embeddings = glove_embeddings
        if self.is_train:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.tokens) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.tokens[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.scores[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.is_train:
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if self.is_train:
            self.tokens, self.scores = shuffle(self.tokens, self.scores, random_state=RAND_SEED)

    def train_generate(self, batch_x, batch_y):
        batch_features = []
        batch_scores = []
        for token, score in zip(batch_x, batch_y):
            try:
                batch_features.append(self.embeddings[token])
                batch_scores.append(score)
            except:
                # If the embeddings can not be found, just append the last entry again
                batch_features.append(batch_features[-1])
                batch_scores.append(batch_scores[-1])
        batch_features = np.array(batch_features, dtype=np.float16)
        batch_scores = np.array(batch_scores, dtype=np.float16)
        return batch_features, batch_scores

    def valid_generate(self, batch_x, batch_y):
        batch_features = []
        batch_scores = []
        for token, score in zip(batch_x, batch_y):
            try:
                batch_features.append(self.embeddings[token])
                batch_scores.append(score)
            except:
                # If the embeddings can not be found, just append the last entry again
                batch_features.append(batch_features[-1])
                batch_scores.append(batch_scores[-1])
        batch_features = np.array(batch_features, dtype=np.float16)
        batch_scores = np.array(batch_scores, dtype=np.float16)
        return batch_features, batch_scores


# BACTH GENERATOR FOR CLASS PREDICTION MODEL
class SentimentClassGenerator(Sequence):
    def __init__(self, tokens, scores, n_classes, class_medians, class_step, glove_embeddings, batch_size, is_train=False):
        self.tokens = tokens
        self.scores = scores
        self.batch_size = batch_size
        self.is_train = is_train
        self.embeddings = glove_embeddings
        self.n_classes = n_classes
        self.class_medians = class_medians
        self.class_step = class_step
        if self.is_train:
            self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.tokens) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.tokens[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.scores[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.is_train:
            return self.train_generate(batch_x, batch_y)
        return self.valid_generate(batch_x, batch_y)

    def on_epoch_end(self):
        if self.is_train:
            self.tokens, self.scores = shuffle(self.tokens, self.scores, random_state=RAND_SEED)

    def train_generate(self, batch_x, batch_y):
        batch_features = []
        batch_classes = []
        for token, score in zip(batch_x, batch_y):
            try:
                batch_features.append(self.embeddings[token])
                score_classes = np.zeros((self.n_classes, ))
                closest_index = np.argmin(np.abs(self.class_medians-score))
                closest_probability = 1 - np.abs(score-self.class_medians[closest_index]) / self.class_step
                score_classes[closest_index] = closest_probability
                if score < self.class_medians[closest_index]:
                    score_classes[closest_index-1] = 1 - closest_probability
                else:
                    score_classes[closest_index+1] = 1 - closest_probability
                batch_classes.append(score_classes)
            except:
                # If the embeddings can not be found, just append the last entry again
                batch_features.append(batch_features[-1])
                batch_classes.append(batch_classes[-1])
        batch_features = np.array(batch_features, dtype=np.float32)
        batch_classes = np.array(batch_classes, dtype=np.float32)
        return batch_features, batch_classes

    def valid_generate(self, batch_x, batch_y):
        batch_features = []
        batch_classes = []
        for token, score in zip(batch_x, batch_y):
            try:
                batch_features.append(self.embeddings[token])
                score_classes = np.zeros((self.n_classes,))
                closest_index = np.argmin(np.abs(self.class_medians - score))
                closest_probability = 1 - np.abs(score - self.class_medians[closest_index]) / self.class_step
                score_classes[closest_index] = closest_probability
                if score < self.class_medians[closest_index]:
                    score_classes[closest_index - 1] = 1 - closest_probability
                else:
                    score_classes[closest_index + 1] = 1 - closest_probability
                batch_classes.append(score_classes)
            except:
                # If the embeddings can not be found, just append the last entry again
                batch_features.append(batch_features[-1])
                batch_classes.append(batch_classes[-1])
        batch_features = np.array(batch_features, dtype=np.float32)
        batch_classes = np.array(batch_classes, dtype=np.float32)
        return batch_features, batch_classes
