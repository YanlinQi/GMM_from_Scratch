# @Author: Yanlin Qi

from mnist.loader import MNIST
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import mode
from sklearn.metrics import confusion_matrix


class GMM:
    def __init__(self, k, model, iter_thrld=0.0001, sigma_s=0.3):
        self.k = k
        self.iter_thrld = iter_thrld
        self.model = model
        self.sigma_s = sigma_s
        self.cur_likelihood = 0.0
        self.likelihood_ls = [0.0]
        self.iter_num = 0

    def initialize(self, X):
        self.shape = X.shape
        self.n, self.m = self.shape

        self.phi = np.full(shape=self.k, fill_value=1 / self.k)
        self.weights = np.full(shape=self.shape, fill_value=1 / self.k)

        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [X[row_index, :].reshape(-1, 1) for row_index in random_row]
        self.sigma = [self.define_sigma(X) for _ in range(self.k)]

    def e_step(self, X):
        """
        # E-Step: update weights and phi holding mu and sigma constant
        """
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)

    def m_step(self, X):
        """
        M-Step: update mu and sigma holding phi and weights constant
        """
        for j in range(self.k):
            sgmj_nomint = np.zeros((X.shape[1], X.shape[1]))
            weight = self.weights[:, [j]]
            total_weight = weight.sum()
            self.mu[j] = ((X * weight).sum(axis=0) / total_weight).reshape(-1, 1)
            """
            # only for Guassian
            self.sigma[i] = np.cov(X.T,
                                   aweights=(weight / total_weight).flatten(),
                                   bias=True)
            """
            for i in range(self.n):     # i in 1,...,n
                x_i = X[i, :].reshape(-1, 1)
                sgmj_nomint = sgmj_nomint + (x_i-self.mu[j])@np.transpose(x_i-self.mu[j])*weight[i, 0]/total_weight

            self.sigma[j] = sgmj_nomint
            self.sigma[j] = self.sigma[j] + 0.05*np.identity((self.sigma[j]).shape[1])

    def fit(self, X):
        self.initialize(X)
        while self.iter_num >= 0:
            self.e_step(X)
            self.m_step(X)
            self.iter_num += 1
            print("iter =", self.iter_num)
            frac_change = (self.cur_likelihood-self.likelihood_ls[-2])/self.likelihood_ls[-1]
            print("frac_change =", frac_change)
            if np.abs(frac_change) < self.iter_thrld:
                print("total iterations:", self.iter_num)
                break

    def predict_proba(self, X):

        ajs = np.zeros((self.n, self.k))
        for j in range(self.k):
            """
            # default pdf function not work -> underflow
            distribution = multivariate_normal(
                           mean=self.mu[j].reshape(-1,),
                           cov=self.sigma[j])

            proba_arr[:, j] = distribution.pdf(X)
            """
            aj = self.pdf(X, mu_j=self.mu[j], sigma_j=self.sigma[j], phi_j=self.phi[j])

            aj = aj.reshape(-1, )
            ajs[:, j] = aj

        max_matrix = np.max(ajs, axis=1).reshape(-1,1) @ np.ones(self.k).reshape(1,-1)
        ajs_minus_max = ajs - max_matrix
        numerator = ajs
        exp_ajs_minus_max = np.exp(ajs_minus_max)
        denominator = np.log(exp_ajs_minus_max.sum(axis=1)[:, np.newaxis]) + np.max(ajs, axis=1).reshape(-1,1)

        self.cur_likelihood = np.sum(denominator)
        self.likelihood_ls.append(np.sum(denominator))

        # F_ij = numerator / denominator
        log_F_ij = numerator - denominator
        F_ij = np.exp(log_F_ij)
        return F_ij

    def pdf(self, X, mu_j, sigma_j, phi_j):
        n, m = X.shape
        guass_exp_ls = []
        diag_vars = sigma_j.diagonal()
        dnmt_sgmj = np.prod(np.sqrt(diag_vars))
        norm_coef = 1 / (np.power(2*np.pi, m/2) * dnmt_sgmj)
        inv_sgmj = np.diag(1/diag_vars)

        for i in range(n):
            x_i = X[i, :].reshape(-1, 1)
            if self.model == "diagonal Gaussian":
                norm_exp_0 = -0.5 * np.transpose(x_i - mu_j) @ inv_sgmj @ (x_i - mu_j)
                norm_exp = norm_exp_0 + np.log(phi_j * norm_coef)
            elif self.model == "spherical Gaussian":
                norm_exp = -0.5 * np.transpose(x_i - mu_j) @ (x_i - mu_j) / diag_vars[0]
            guass_exp_ls.append(norm_exp)

        guass_exp_ls = np.asarray(guass_exp_ls)
        prob = guass_exp_ls.reshape(-1, 1)

        return prob

    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)

    def define_sigma(self, X):
        scale_factor = 1e-02
        if self.model == "diagonal Gaussian":
            cov_x = np.random.rand(X.shape[1]) * np.identity(X.shape[1]) * scale_factor

        elif self.model == "spherical Gaussian":
            cov_x = self.sigma_s * np.identity(X.shape[1]) * scale_factor
        elif self.model == "Gaussian":
            cov_x = np.cov(X.T) * scale_factor
        else:
            print("The variance of the model cannot be calculated...")
            return
        return cov_x


def rebin(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def main():
    imsize = 28
    dscale_size = int(imsize/2)
    mndata = MNIST('./mnist')
    model_ls = ["diagonal Gaussian", "spherical Gaussian", "Gaussian"]
    cluster_ls = [0, 1, 2, 3, 4]

    trn_imgs, trn_labs = mndata.load_training()
    trn_imgs = list(map(lambda x: rebin(np.array(x).reshape(imsize, imsize), (dscale_size, dscale_size)), trn_imgs))
    trn_imgs = list(map(lambda x: x.reshape(-1, 1), trn_imgs))

    res = [(trn_imgs[idx], _) for idx, _ in enumerate(trn_labs) if _ in cluster_ls]
    img_vecs = list(map(lambda x: x[0], res))
    img_vecs = np.array(img_vecs).reshape(len(img_vecs), -1)
    labs = list(map(lambda x: x[1], res))

    # np.random.seed(42)
    gmm = GMM(k=5, iter_thrld=0.0001, model=model_ls[0])
    gmm.fit(img_vecs)

    lab_arr = np.array(labs).reshape(-1, 1)
    permutation = np.array([mode(lab_arr[gmm.predict(img_vecs).reshape(-1,1) == i]).mode.item()
                            for i in range(gmm.k)])

    permuted_prediction = permutation[gmm.predict(img_vecs)]
    print(np.mean(labs == permuted_prediction))

    # ########################## 3.(ii) ##########################
    # calculate the confusion matrix
    cm = confusion_matrix(labs, permuted_prediction)
    err_counts = img_vecs.shape[0] - np.sum(np.diagonal(cm))
    err_rate = err_counts/img_vecs.shape[0]
    print("The error rate is:", err_rate)


if __name__ == '__main__':
    main()
