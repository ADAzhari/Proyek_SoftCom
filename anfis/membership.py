import numpy as np

class GaussianMF:
    """
    Gaussian Membership Function
    Paling umum dipakai di ANFIS karena smooth & differentiable
    (penting untuk backpropagation)
    
    f(x) = exp(-((x - mean)^2) / (2 * sigma^2))
    """
    def __init__(self, mean, sigma):
        self.mean  = mean
        self.sigma = sigma

    def compute(self, x):
        return np.exp(-((x - self.mean) ** 2) / (2 * self.sigma ** 2 + 1e-8))

    def grad_mean(self, x):
        """gradient terhadap mean — untuk backprop"""
        mu = self.compute(x)
        return mu * (x - self.mean) / (self.sigma ** 2 + 1e-8)

    def grad_sigma(self, x):
        """gradient terhadap sigma — untuk backprop"""
        mu = self.compute(x)
        return mu * ((x - self.mean) ** 2) / (self.sigma ** 3 + 1e-8)


class FuzzyVariable:
    """
    Satu variabel input dengan beberapa membership function
    Contoh: EAR punya 3 MF → 'rendah', 'sedang', 'tinggi'
    """
    def __init__(self, name, n_mf=3, range_min=0.0, range_max=1.0):
        self.name    = name
        self.n_mf    = n_mf

        # inisialisasi mean & sigma secara merata di range input
        means  = np.linspace(range_min, range_max, n_mf)
        sigma  = (range_max - range_min) / (n_mf * 2)
        sigmas = np.full(n_mf, sigma)

        self.mfs = [GaussianMF(m, s) for m, s in zip(means, sigmas)]

    def fuzzify(self, x):
        """
        Input  : nilai crisp (angka)
        Output : array derajat keanggotaan, shape (n_mf,)
        """
        return np.array([mf.compute(x) for mf in self.mfs])

    def get_params(self):
        means  = [mf.mean  for mf in self.mfs]
        sigmas = [mf.sigma for mf in self.mfs]
        return np.array(means), np.array(sigmas)

    def set_params(self, means, sigmas):
        for mf, m, s in zip(self.mfs, means, sigmas):
            mf.mean  = m
            mf.sigma = s