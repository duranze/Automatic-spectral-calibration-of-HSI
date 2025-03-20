import numpy as np
from sklearn.cluster import MiniBatchKMeans

def computeRMSE(groundTruth, recovered):
    """
    Compute Root Mean Squared Error (RMSE) between two images.
    :param groundTruth: Ground truth image (Height x Width x Spectral Dimension)
    :param recovered: Reconstructed image (Height x Width x Spectral Dimension)
    :return: RMSE value
    """
    assert groundTruth.shape == recovered.shape, "Mismatch in image dimensions."
    difference = (groundTruth - recovered) ** 2
    return np.sqrt(np.mean(difference))

def calc_ergas(img_tgt, img_fus):
    """
    Compute ERGAS metric.
    :param img_tgt: Target image
    :param img_fus: Fused/Reconstructed image
    :return: ERGAS value
    """
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    rmse = np.sqrt(np.mean((img_tgt - img_fus) ** 2, axis=1))
    mean_val = np.mean(img_tgt, axis=1)
    ergas = np.mean((rmse / mean_val) ** 2)
    return 100 / 4 * ergas ** 0.5

def calc_psnr(img_tgt, img_fus):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    :param img_tgt: Target image
    :param img_fus: Fused/Reconstructed image
    :return: PSNR value
    """
    mse = np.mean((img_tgt - img_fus) ** 2)
    img_max = np.max(img_tgt)
    return 10 * np.log10(img_max ** 2 / mse)

def calc_rmse(img_tgt, img_fus):
    """
    Compute Root Mean Squared Error (RMSE).
    :param img_tgt: Target image
    :param img_fus: Fused/Reconstructed image
    :return: RMSE value
    """
    return np.sqrt(np.mean((img_tgt - img_fus) ** 2))

def calc_sam(img_tgt, img_fus):
    """
    Compute Spectral Angle Mapper (SAM).
    :param img_tgt: Target image
    :param img_fus: Fused/Reconstructed image
    :return: SAM value in degrees
    """
    img_fus[img_tgt == 0] = 1e-4
    img_tgt[img_tgt == 0] = 1e-4
    img_tgt = np.squeeze(img_tgt)
    img_fus = np.squeeze(img_fus)
    img_tgt = img_tgt.reshape(img_tgt.shape[0], -1)
    img_fus = img_fus.reshape(img_fus.shape[0], -1)
    img_tgt = img_tgt / np.max(img_tgt)
    img_fus = img_fus / np.max(img_fus)
    A = np.sqrt(np.sum(img_tgt ** 2, axis=0))
    B = np.sqrt(np.sum(img_fus ** 2, axis=0))
    AB = np.sum(img_tgt * img_fus, axis=0)
    sam = AB / (A * B)
    sam = np.clip(sam, -1.0, 1.0)
    sam = np.arccos(sam)
    return np.mean(sam) * 180 / np.pi




