import numpy as np
from scipy import stats


def diebold_mariano_test(y_true, pred_1, pred_2):
    """
    Diebold-Mariano test for comparing forecast accuracy.

    pred_1 = predictions from model 1
    pred_2 = predictions from model 2

    Lower error is better.
    """

    y_true = np.array(y_true)
    pred_1 = np.array(pred_1)
    pred_2 = np.array(pred_2)

    error_1 = (y_true - pred_1) ** 2
    error_2 = (y_true - pred_2) ** 2

    loss_diff = error_1 - error_2

    dm_stat = np.mean(loss_diff) / (np.std(loss_diff, ddof=1) / np.sqrt(len(loss_diff)))

    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=len(loss_diff) - 1))

    return dm_stat, p_value