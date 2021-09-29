"""Normalize features"""

import numpy as np


def normalize(features):

    features_normalized = np.copy(features).astype(float)

    # Calculate mean
    features_mean = np.mean(features, 0)

    # Calculate standard deviation
    features_deviation = np.std(features, 0)

    # Standardized operation
    if features.shape[0] > 1:
        features_normalized -= features_mean

    # Avoid division by zero
    features_deviation[features_deviation == 0] = 1
    features_normalized /= features_deviation

    return features_normalized, features_mean, features_deviation
