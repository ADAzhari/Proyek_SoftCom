import numpy as np
import pickle
from anfis.anfis_model import ANFIS

class ANFISTrainer:
    """
    Training ANFIS dengan hybrid learning:
    - Consequent params → Least Squares Estimation (LSE)
    - Premise params (mean/sigma MF) → Gradient Descent
    """

    def __init__(self, model: ANFIS, lr=0.01):
        self.model = model
        self.lr    = lr

    def _build_phi(self, X):
        """
        Bangun matrix phi untuk LSE consequent update
        shape: (n_samples, n_rules * (n_inputs + 1))
        """
        n_samples = X.shape[0]
        n_cols    = self.model.n_rules * (self.model.n_inputs + 1)
        phi       = np.zeros((n_samples, n_cols))

        for i, x in enumerate(X):
            mu_list = self.model.layer1_fuzzify(x)
            w       = self.model.layer2_fire_rules(mu_list)
            w_norm  = self.model.layer3_normalize(w)
            x_aug   = np.append(x, 1.0)

            for r in range(self.model.n_rules):
                start = r * (self.model.n_inputs + 1)
                end   = start + (self.model.n_inputs + 1)
                phi[i, start:end] = w_norm[r] * x_aug

        return phi

    def update_consequents_lse(self, X, y):
        """
        Least Squares: theta = (phi^T phi)^-1 phi^T y
        """
        phi   = self._build_phi(X)
        theta, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
        self.model.consequents = theta.reshape(
            self.model.n_rules, self.model.n_inputs + 1
        )

    def update_premises_gd(self, X, y):
        """
        Gradient descent update untuk mean & sigma MF
        """
        for x, target in zip(X, y):
            output  = self.model.forward(x)
            error   = output - target

            mu_list = self.model.layer1_fuzzify(x)
            w       = self.model.layer2_fire_rules(mu_list)
            w_norm  = self.model.layer3_normalize(w)
            x_aug   = np.append(x, 1.0)

            f_rules = self.model.consequents @ x_aug
            w_sum   = w.sum() + 1e-8

            for j, var in enumerate(self.model.variables):
                d_means  = np.zeros(self.model.n_mf)
                d_sigmas = np.zeros(self.model.n_mf)

                for k in range(self.model.n_mf):
                    d_output_d_wk = 0.0
                    for r, rule in enumerate(self.model.rules):
                        if rule[j] == k:
                            # chain rule melalui normalisasi
                            d_wnorm_d_wk = (w_sum - w[r]) / (w_sum ** 2)
                            d_output_d_wk += f_rules[r] * d_wnorm_d_wk

                    mf = var.mfs[k]
                    xj = x[j]
                    d_means[k]  = error * d_output_d_wk * mf.grad_mean(xj)
                    d_sigmas[k] = error * d_output_d_wk * mf.grad_sigma(xj)

                means, sigmas = var.get_params()
                means  -= self.lr * d_means
                sigmas -= self.lr * d_sigmas
                sigmas  = np.maximum(sigmas, 1e-4)  # sigma tidak boleh negatif
                var.set_params(means, sigmas)

    def train(self, X, y, epochs=50, verbose=True):
        """
        Hybrid learning loop
        """
        history = []

        for epoch in range(epochs):
            # Step 1: update consequents dengan LSE (forward pass)
            self.update_consequents_lse(X, y)

            # Step 2: update premises dengan GD (backward pass)
            self.update_premises_gd(X, y)

            # hitung loss
            y_pred = self.model.predict(X)
            mse    = np.mean((y_pred - y) ** 2)
            rmse   = np.sqrt(mse)
            history.append(rmse)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | RMSE: {rmse:.4f}")

        return history

    def save_model(self, path="models/anfis_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        print(f"Model saved → {path}")

    @staticmethod
    def load_model(path="models/anfis_model.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Model loaded ← {path}")
        return model