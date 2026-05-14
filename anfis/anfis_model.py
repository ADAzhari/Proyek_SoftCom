import numpy as np
from itertools import product
from anfis.membership import FuzzyVariable

class ANFIS:
    """
    Adaptive Neuro-Fuzzy Inference System
    
    Input  : 5 fitur [EAR, EAR_asym, PERCLOS, BROW, PITCH]
    Output : Fatigue Score (0-100)
    
    Arsitektur 5 layer:
    L1 - Fuzzifikasi
    L2 - Rule firing (product operator)
    L3 - Normalisasi
    L4 - Defuzzifikasi (consequent linear)
    L5 - Output agregasi
    """

    def __init__(self, n_mf=3):
        self.n_mf = n_mf

        # definisi variabel input + range nilainya
        self.variables = [
            FuzzyVariable("EAR",      n_mf, 0.10, 0.40),
            FuzzyVariable("EAR_asym", n_mf, 0.00, 0.20),
            FuzzyVariable("PERCLOS",  n_mf, 0.00, 1.00),
            FuzzyVariable("BROW",     n_mf, 0.00, 1.00),
            FuzzyVariable("PITCH",    n_mf, -30,  30.00),
        ]

        self.n_inputs = len(self.variables)

        # generate semua kombinasi rule
        # 5 input × 3 MF = 3^5 = 243 rules
        self.rules = list(product(range(n_mf), repeat=self.n_inputs))
        self.n_rules = len(self.rules)

        # consequent parameters (linear: p*x + q per rule)
        # shape: (n_rules, n_inputs + 1) — +1 untuk bias
        self.consequents = np.random.randn(
            self.n_rules, self.n_inputs + 1
        ) * 0.01

    # -------------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------------

    def layer1_fuzzify(self, x):
        """
        Input  : x shape (n_inputs,)
        Output : list of arrays, tiap array shape (n_mf,)
        """
        return [var.fuzzify(xi) for var, xi in zip(self.variables, x)]

    def layer2_fire_rules(self, mu_list):
        """
        Hitung firing strength tiap rule
        pakai product T-norm
        Output : array shape (n_rules,)
        """
        w = np.ones(self.n_rules)
        for i, rule in enumerate(self.rules):
            for j, mf_idx in enumerate(rule):
                w[i] *= mu_list[j][mf_idx]
        return w

    def layer3_normalize(self, w):
        """
        Normalisasi firing strength
        Output : array shape (n_rules,)
        """
        total = w.sum() + 1e-8
        return w / total

    def layer4_consequent(self, w_norm, x):
        """
        Hitung output tiap rule (linear Sugeno)
        f_i = p_i1*x1 + p_i2*x2 + ... + p_in*xn + q_i
        Output : array shape (n_rules,)
        """
        x_aug = np.append(x, 1.0)  # tambah bias
        f = self.consequents @ x_aug  # shape (n_rules,)
        return w_norm * f

    def layer5_output(self, wf):
        """
        Agregasi output akhir → Fatigue Score mentah
        """
        return wf.sum()

    def forward(self, x):
        """
        Full forward pass
        Input  : x shape (n_inputs,)
        Output : fatigue score (float)
        """
        mu_list = self.layer1_fuzzify(x)
        w       = self.layer2_fire_rules(mu_list)
        w_norm  = self.layer3_normalize(w)
        wf      = self.layer4_consequent(w_norm, x)
        output  = self.layer5_output(wf)

        # clamp ke range 0-100
        return float(np.clip(output, 0, 100))

    def predict(self, X):
        """
        Batch prediction
        Input  : X shape (n_samples, n_inputs)
        Output : array shape (n_samples,)
        """
        return np.array([self.forward(x) for x in X])