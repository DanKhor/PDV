import numpy as np


def get_optimal_threshold(model, X, y):
    """
    Get optimal target binarization threshold.
    Balanced accuracy metric is used.

    Parameters
    ----------

    X : numpy.ndarray
    y : numpy.ndarray
      1d vector, target values

    Returns
    -------
    : float
      Chosen threshold.
    """

    scores = model.predict(X)[:, 0]

    # for each score store real targets that correspond score
    score_to_y = dict()
    score_to_y[min(scores) - 1e-5] = [0, 0]
    for one_score, one_y in zip(scores, y):
        score_to_y.setdefault(one_score, [0, 0])
        score_to_y[one_score][one_y] += 1

    # ith element of cum_sums is amount of y <= alpha
    scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
    cum_sums = np.array(y_counts).cumsum(axis=0)

    # count balanced accuracy for each threshold
    recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
    recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
    ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
    best_score = scores[np.argmax(ba_accuracy_values)]
    return best_score