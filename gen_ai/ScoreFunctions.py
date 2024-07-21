import numpy as np
import pandas as pd
from scipy.special import softmax

def sentiment_score(scores):
    """
    This function takes a list of sentiment scores, calculates the softmax of
    the scores, and identifies the sentiment with the highest probability.
    It then clips the score to ensure it falls within predefined bounds
    for positive, negative, and neutral sentiments.
    
    Original Bound -> OB
    Desired Bound -> DB

    new score = DB[lower] + (DB[upper] - DB[lower]) * (current score - OB[lower])
                             ---------------------
                            (OB[upper] - OB[lower])

    note: 
    There is some discontinuity in the calculation of the scores.
    for example:
        for a neutral label with sentiment score of -0.33 it mean there is no positive influence.
        but for negative label with negative score of -0.92 there might be some positive influence.

    """
    def clip_score(original_score, desired_bound, original_bound):
        """."""
        normalized_difference = (original_score - original_bound['lower'])\
            / (original_bound['upper'] - original_bound['lower'])
        score = desired_bound['lower'] + normalized_difference *\
            (desired_bound['upper'] - desired_bound['lower'])
        return score

    softmax_scores = softmax(scores)
    max_ind = softmax_scores.argmax()

    neutral_desired_bound = {'lower': -0.20, 'upper': 0.20}
    positive_desired_bound = {'lower': neutral_desired_bound['upper']+0.01, 'upper': 1}
    negative_desired_bound = {'lower': -1, 'upper': neutral_desired_bound['lower']-0.01}

    if max_ind == 2:
        pos_score = softmax_scores.max()
        score = clip_score(pos_score, positive_desired_bound, {'lower': 0.33, 'upper': 1})
    elif max_ind == 0:
        neg_score = -1*softmax_scores.max()
        score = clip_score(neg_score, negative_desired_bound, {'lower': -1, 'upper': -0.33})
    else:
        neutral_score = softmax_scores[2]-softmax_scores[0]
        score = clip_score(neutral_score, neutral_desired_bound, {'lower': -0.49, 'upper': 0.49})

    return score