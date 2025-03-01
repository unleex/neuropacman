import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing as ppc 
from scipy import signal as sig
from scipy.stats import entropy
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
import json
from sklearn.base import BaseEstimator, TransformerMixin


def to_channels(data):
    data = data[:len(data) - len(data) % 6]
    return np.array([data[i::6] for i in range(6)])


def from_channels(channels):
    return channels.reshape(np.prod(channels.shape))


def from_channels_matrix(data):
    return np.array([from_channels(channels) for channels in data])


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig. butter(order, [low, high], btype= 'band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass (lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


def channelwise_bandpass_filtering(data, lowcut, highcut, fs, order=5):
    channels = to_channels(data)
    filtered = []
    for ch in channels:
        filtered.append(butter_bandpass_filter(ch, lowcut, highcut, fs, order))
    return np.array(filtered)


def baseline_correction(channel, baseline_length):
    correction = channel[:baseline_length].mean()
    return channel - correction


def channelwise_baseline_correction(data, baseline_length):
    channels = to_channels(data)
    return np.array([baseline_correction(channel, baseline_length) for channel in channels])


def maxon_preprocess(
    data: pd.DataFrame, fs, 
    baseline_length, 
    filter_lowcut, 
    filter_highcut, 
    filter_order, ):
    data = [channelwise_baseline_correction(channels, baseline_length) for channels in data]
    data = [channelwise_bandpass_filtering(channels, filter_lowcut, filter_highcut, fs, filter_order) for channels in data]
    return np.array(data).squeeze(2)



class EEGPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, fs=250, baseline_length=30, 
                 filter_lowcut=0.1, filter_highcut=40, 
                 filter_order=5):
        self.fs = fs
        self.baseline_length = baseline_length
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        self.filter_order = filter_order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed = maxon_preprocess(
            X, fs=self.fs, 
            baseline_length=self.baseline_length, 
            filter_lowcut=self.filter_lowcut, 
            filter_highcut=self.filter_highcut, 
            filter_order=self.filter_order, 
        )
        return from_channels_matrix(processed)
    

    

class Scaler:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, x, y=None):
        x = x.to_numpy()
        self.scaler.fit(x)
        return self.scaler
        
    def transform(self, x):
        return pd.DataFrame(self.scaler.transform(x.to_numpy()))

params = {"baseline_length": 28, "filter_lowcut": 0.11246175834572819, "filter_highcut": 10.552734135774276, "filter_order": 4}

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ('preprocessing', EEGPreprocessor(**params)),
    ('model', LogisticRegression())
])


# data = pd.read_csv('train_10.csv', sep=';')
# X_raw = data.drop("class", axis=1)
# y = data["class"]

# from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X_raw, y, test_size=0.1, random_state=69)

# pipeline.fit(X_train, y_train)
# predictions = pipeline.predict(X_test)

# print(metric(y_test, predictions))
