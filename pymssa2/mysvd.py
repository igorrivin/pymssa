import torch
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import randomized_svd
import time

def randomized_svd_gpu(M, n_components, n_oversamples=10, n_iter='auto',
                       transpose='auto', random_state=0, lib='pytorch',tocpu=True):
    """Computes a truncated randomized SVD on GPU. Adapted from Sklearn.

    Parameters
    ----------
    M : ndarray or sparse matrix
        Matrix to decompose
    n_components : int
        Number of singular values and vectors to extract.
    n_oversamples : int (default is 10)
        Additional number of random vectors to sample the range of M so as
        to ensure proper conditioning. The total number of random vectors
        used to find the range of M is n_components + n_oversamples. Smaller
        number can improve speed but can negatively impact the quality of
        approximation of singular vectors and singular values.
    n_iter : int or 'auto' (default is 'auto')
        Number of power iterations. It can be used to deal with very noisy
        problems. When 'auto', it is set to 4, unless `n_components` is small
        (< .1 * min(X.shape)) `n_iter` in which case is set to 7.
        This improves precision with few components.
    transpose : True, False or 'auto' (default)
        Whether the algorithm should be applied to M.T instead of M. The
        result should approximately be the same. The 'auto' mode will
        trigger the transposition if M.shape[1] > M.shape[0] since this
        implementation of randomized SVD tend to be a little faster in that
        case.
    random_state : int, RandomState instance or None, optional (default=None)
        The seed of the pseudo random number generator to use when shuffling
        the data.  If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by `np.random`.
    lib : {'cupy', 'pytorch'}, str optional
        Chooses the GPU library to be used.

    Notes
    -----
    This algorithm finds a (usually very good) approximate truncated
    singular value decomposition using randomization to speed up the
    computations. It is particularly fast on large matrices on which
    you wish to extract only a small number of components. In order to
    obtain further speed up, `n_iter` can be set <=2 (at the cost of
    loss of precision).

    References
    ----------
    * Finding structure with randomness: Stochastic algorithms for constructing
      approximate matrix decompositions
      Halko, et al., 2009 http://arxiv.org/abs/arXiv:0909.4061
    * A randomized algorithm for the decomposition of matrices
      Per-Gunnar Martinsson, Vladimir Rokhlin and Mark Tygert
    * An implementation of a randomized algorithm for principal component
      analysis
      A. Szlam et al. 2014
    """
    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == 'auto':
        # Checks if the number of iterations is explicitly specified
        n_iter = 7 if n_components < .1 * min(M.shape) else 4

    if transpose == 'auto':
        transpose = n_samples < n_features
    if transpose:
        M = M.T # this implementation is a bit faster with smaller shape[1]

    if lib == 'pytorch':
        M_gpu = torch.Tensor.cuda(torch.from_numpy(M.astype('float32')))

        # Generating normal random vectors with shape: (M.shape[1], n_random)
        Q = torch.cuda.FloatTensor(M_gpu.shape[1], n_random).normal_()

        # Perform power iterations with Q to further 'imprint' the top
        # singular vectors of M in Q
        for i in range(n_iter):
            Q = torch.mm(M_gpu, Q)
            Q = torch.mm(torch.transpose(M_gpu, 0, 1), Q)

        # Sample the range of M using by linear projection of Q. Extract an orthonormal basis
        Q, _ = torch.qr(torch.mm(M_gpu, Q))

        # project M to the (k + p) dimensional space using the basis vectors
        B = torch.mm(torch.transpose(Q, 0, 1), M_gpu)

        # compute the SVD on the thin matrix: (k + p) wide
        Uhat, s, V = torch.svd(B)
        del B
        U = torch.mm(Q, Uhat)

        if transpose:
            # transpose back the results according to the input convention
            U, s, V=(torch.transpose(V[:n_components, :], 0, 1),s[:n_components],torch.transpose(U[:, :n_components], 0, 1))
        else:
            U, s, V=( U[:, :n_components], s[:n_components], V[:n_components, :])
        
        if tocpu is True:
            return np.array(U.cpu()), np.array(s.cpu()), np.array(V.cpu())
        else:
            return U, s, V

def tester(func, m, n, k, numiter):
    ar = np.random.randn(m, n)
    t1 = time.time()
    for i in range(numiter):
        a, b, c = func(ar, k)
    t2 = time.time()
    return t2-t1
    