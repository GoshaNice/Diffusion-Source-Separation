import editdistance
import numpy as np

# Don't forget to support cases when target_text == ''


def calc_cer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text != "":
            return 1
        return 0
    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    if target_text == "":
        if predicted_text != "":
            return 1
        return 0
    target_text_splitted = target_text.split(" ")
    predicted_text_splitted = predicted_text.split(" ")
    return editdistance.eval(target_text_splitted, predicted_text_splitted) / len(
        target_text_splitted
    )


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
