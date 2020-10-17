import cv2
import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg

# import matplotlib.pyplot as plt

from skimage import feature

def make_system(L, sparse_map, constraint_factor=0.001):
    # split trimap into foreground, background, known and unknown masks
    spflatten = sparse_map.ravel()

    D = scipy.sparse.diags(spflatten)

    # combine constraints and graph laplacian
    A = constraint_factor * D + L
    # constrained values of known alpha values
    b = constraint_factor * D * spflatten

    return A, b


def g1x(x, y, s1):
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(x, 2 * np.pi * s1sq ** 2),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))

    return g


def g1y(x, y, s1):
    s1sq = s1 ** 2
    g = -1 * np.multiply(np.divide(y, 2 * np.pi * s1sq ** 2),
                         np.exp(-1 * np.divide(x ** 2 + y ** 2, 2 * s1sq)))

    return g


def estimate_sparse_blur(gimg, edge_map, std1, std2):
    half_window = 11
    m = half_window * 2 + 1
    a = np.arange(-half_window, half_window + 1)
    xmesh = np.tile(a, (m, 1))
    ymesh = xmesh.T

    f11 = g1x(xmesh, ymesh, std1)
    f12 = g1y(xmesh, ymesh, std1)

    f21 = g1x(xmesh, ymesh, std2)
    f22 = g1y(xmesh, ymesh, std2)

    gimx1 = scipy.ndimage.convolve(gimg, f11, mode='nearest')
    gimy1 = scipy.ndimage.convolve(gimg, f12, mode='nearest')
    mg1 = np.sqrt(gimx1 ** 2 + gimy1 ** 2)

    gimx2 = scipy.ndimage.convolve(gimg, f21, mode='nearest')
    gimy2 = scipy.ndimage.convolve(gimg, f22, mode='nearest')
    mg2 = np.sqrt(gimx2 ** 2 + gimy2 ** 2)

    R = np.divide(mg1, mg2)
    R = np.multiply(R, edge_map > 0)

    sparse_bmap = np.sqrt(np.divide(R ** 2 * (std1 ** 2) - (std2 ** 2), 1 - R ** 2))
    sparse_bmap[np.isnan(sparse_bmap)] = 0
    sparse_bmap[sparse_bmap > 5] = 5

    return sparse_bmap


def get_laplacian(I, r=1):
    eps = 0.0000001
    h, w, c = I.shape
    wr = (2 * r + 1) * (2 * r + 1)

    M_idx = np.arange(h * w).reshape(w, h).T
    n_vals = (w - 2 * r) * (h - 2 * r) * wr ** 2

    # data for matting laplacian in coordinate form
    row_idx = np.zeros(n_vals, dtype=np.int64)
    col_idx = np.zeros(n_vals, dtype=np.int64)
    vals = np.zeros(n_vals, dtype=np.float64)
    lenr = 0

    for j in range(r, h - r):
        for i in range(r, w - r):
            winr = I[j - r:j + r + 1, i - r:i + r + 1, 2]
            wing = I[j - r:j + r + 1, i - r:i + r + 1, 1]
            winb = I[j - r:j + r + 1, i - r:i + r + 1, 0]
            win_idx = M_idx[j - r:j + r + 1, i - r:i + r + 1].T.ravel()

            meanwinr = winr.mean()
            winrsq = np.multiply(winr, winr)
            varI_rr = winrsq.sum() / wr - meanwinr ** 2

            meanwing = wing.mean()
            wingsq = np.multiply(wing, wing)
            varI_gg = wingsq.sum() / wr - meanwing ** 2

            meanwinb = winb.mean()
            winbsq = np.multiply(winb, winb)
            varI_bb = winbsq.sum() / wr - meanwinb ** 2

            winrgsq = np.multiply(winr, wing)
            varI_rg = winrgsq.sum() / wr - meanwinr * meanwing

            winrbsq = np.multiply(winr, winb)
            varI_rb = winrbsq.sum() / wr - meanwinr * meanwinb

            wingbsq = np.multiply(wing, winb)
            varI_gb = wingbsq.sum() / wr - meanwing * meanwinb

            Sigma = np.array([[varI_rr, varI_rg, varI_rb],
                              [varI_rg, varI_gg, varI_gb],
                              [varI_rb, varI_gb, varI_bb]])

            meanI = np.array([meanwinr, meanwing, meanwinb])

            Sigma = Sigma + eps * np.eye(3)

            winI = np.zeros((wr, c))

            winI[:, 0] = winr.T.ravel()
            winI[:, 1] = wing.T.ravel()
            winI[:, 2] = winb.T.ravel()

            winI = winI - meanI

            inv_cov = np.linalg.inv(Sigma)
            tvals = (1 + np.matmul(np.matmul(winI, inv_cov), winI.T)) / wr

            row_idx[lenr:wr ** 2 + lenr] = np.tile(win_idx, (1, wr)).ravel()
            col_idx[lenr:wr ** 2 + lenr] = np.tile(win_idx, (wr, 1)).T.ravel()
            vals[lenr:wr ** 2 + lenr] = tvals.T.ravel()

            lenr += wr ** 2

    # Lsparse = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(h*w, h*w))
    Lsparse = scipy.sparse.coo_matrix((vals, (row_idx, col_idx)), shape=(w * h, w * h))

    row_idx2 = np.zeros(w * h, dtype=np.int64)
    col_idx2 = np.zeros(w * h, dtype=np.int64)
    vals2 = np.zeros(w * h, dtype=np.float64)

    row_idx2[:] = np.arange(w * h)
    col_idx2[:] = np.arange(w * h)
    vals2[:] = Lsparse.sum(axis=1).ravel()

    LDsparse = scipy.sparse.coo_matrix((vals2, (row_idx2, col_idx2)), shape=(w * h, w * h))

    return LDsparse - Lsparse


def estimate_bmap_laplacian(img, sigma_c, std1, std2):
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    edge_map = feature.canny(gimg, sigma_c)

    sparse_bmap = estimate_sparse_blur(gimg, edge_map, std1, std2)
    h, w = sparse_bmap.shape

    L1 = get_laplacian(img / 255.0)
    A, b = make_system(L1, sparse_bmap.T)

    bmap = scipy.sparse.linalg.spsolve(A, b).reshape(w, h).T

    return bmap


def propagate_laplacian(img, bmap):

    L1 = get_laplacian(img / 255.0)

    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
    mask = feature.canny(gimg, 1.0) * 1.0
    h, w = mask.shape

    maskcoord = np.where(mask == 1.0)
    mask[maskcoord[0], maskcoord[1]] = bmap[maskcoord[0], maskcoord[1]]
    A, b = make_system(L1, mask.T)

    bmapLaplacian = scipy.sparse.linalg.spsolve(A, b).reshape(w, h).T

    return bmapLaplacian


