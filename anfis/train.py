import numpy as np
import pickle
from anfis.anfis_model import ANFIS

class ANFISTrainer:
    """
    Training ANFIS dengan pure Gradient Descent.
    Update semua parameter (premise MF + consequent) via backpropagation.
    """

    def __init__(self, model: ANFIS, lr=0.01):
        self.model = model
        self.lr    = lr

    def _train_step(self, x, target):
        """
        Satu langkah forward + backward untuk satu sampel.
        Return: squared error
        """
        # --- Forward ---
        mu_list = self.model.layer1_fuzzify(x)
        w       = self.model.layer2_fire_rules(mu_list)
        w_sum   = w.sum() + 1e-8
        w_norm  = w / w_sum
        x_aug   = np.append(x, 1.0)
        f_rules = self.model.consequents @ x_aug       # (n_rules,)
        output  = np.sum(w_norm * f_rules)
        output_clipped = np.clip(output, 0, 1)
        error   = output_clipped - target

        # Skip jika gradient = 0 karena clipping
        if (output <= 0 and error < 0) or (output >= 1 and error > 0):
            return error ** 2

        # --- Backward: update consequent params (GD) ---
        # dL/d_p[r,j] = error * w_norm[r] * x_aug[j]
        grad_consequents = error * np.outer(w_norm, x_aug)
        self.model.consequents -= self.lr * grad_consequents

        # --- Backward: update premise params (MF mean & sigma) ---
        for j, var in enumerate(self.model.variables):
            xj    = x[j]
            mu_jk = mu_list[j]  # shape (n_mf,)

            for k in range(self.model.n_mf):
                # dOutput/dmu_jk melalui semua rule yang pakai MF k di input j
                d_output_d_mu = 0.0

                for r, rule in enumerate(self.rules_cache):
                    if rule[j] != k:
                        continue

                    # dOutput/dw_r = (f_r - output) / w_sum
                    d_out_d_wr = (f_rules[r] - output) / w_sum

                    # dw_r/dmu_jk = w_r / mu_jk
                    if mu_jk[k] > 1e-10:
                        d_wr_d_mu = w[r] / mu_jk[k]
                    else:
                        d_wr_d_mu = 0.0

                    d_output_d_mu += d_out_d_wr * d_wr_d_mu

                # Chain rule ke MF params
                mf = var.mfs[k]
                grad_mean  = error * d_output_d_mu * mf.grad_mean(xj)
                grad_sigma = error * d_output_d_mu * mf.grad_sigma(xj)

                mf.mean  -= self.lr * grad_mean
                mf.sigma  = max(mf.sigma - self.lr * grad_sigma, 1e-4)

        return error ** 2

    def train(self, X, y, epochs=50, verbose=True):
        """
        Pure gradient descent training loop.
        """
        # Cache rules list untuk akses cepat
        self.rules_cache = self.model.rules

        history = []
        n = len(X)

        for epoch in range(epochs):
            # Shuffle data tiap epoch
            indices = np.random.permutation(n)
            total_loss = 0.0

            for idx in indices:
                total_loss += self._train_step(X[idx], y[idx])

            rmse = np.sqrt(total_loss / n)
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