import numpy as np


def calc_snr(est, target):
    return 20 * np.log10(
        np.linalg.norm(target) / (np.linalg.norm(target - est) + 1e-6) + 1e-6
    )


def calc_si_sdr(est, target):
    alpha = (target * est).sum() / np.linalg.norm(target) ** 2
    return 20 * np.log10(
        np.linalg.norm(alpha * target) / (np.linalg.norm(alpha * target - est) + 1e-6)
        + 1e-6
    )
