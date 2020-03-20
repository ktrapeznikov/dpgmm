
import numpy as np
from scipy.special import softmax, digamma
import tqdm




class DPMixture:
    def __init__(self, alpha = 1, obs_hparams = None, obs_dist_type = None, num_components = 10):
        self.alpha = alpha #beta prior
        self.obs_hparams = obs_hparams #obs prior
        self.num_components = num_components #limit of number of components
        self.obs_hparams = obs_hparams
        self.obs_dist = obs_dist_type(**obs_dist_type, num_components = num_components)

    def initialize_variational_dpmixture(self):
        self.gamma = np.random.beta(self.num_components-1,2) #beta stick priors

    def initialize_variational_data_assignment(self,num_points: int = None):
        return np.random.dirichlet(np.ones(self.num_components),size = num_points)

    def update_variational_dpmixture(self, phi: np.ndarray):
        self.gamma[:,0] = 1. + phi.sum(0)[:-1]
        q_zn_gt_i = self.compute_zn_gt_i(phi)
        self.gamma[:,1] = self.alpha + q_zn_gt_i.sum(0)

    def compute_zn_gt_i(self, phi):
        return np.cumsum(phi[:,::-1],axis=1)[::-1][1:] 

    def update_variational_data_assignments(self, component_resp,  data_resp):
        return softmax(component_resp, data_resp)

    def expected_component_resp(self):
        E_log_V, E_log_1_min_V = self.compute_stick_log_expectations()

        out = np.zeros(self.num_components)
        out[:-1] = E_log_V
        out[1:] += E_log_1_min_V.cumsum()

        return out

    def compute_stick_log_expectations(self):
        digamma_gamma_1_plus_2 = digamma(self.gamma.sum(1))
        digamma_gamma_1 = digamma(self.gamma[:,0])
        digamma_gamma_2 = digamma(self.gamma[:,1])

        E_log_V = digamma_gamma_1 - digamma_gamma_1_plus_2
        E_log_1_min_V = digamma_gamma_2 - digamma_gamma_1_plus_2

        return E_log_V, E_log_1_min_V


    def fit(self, X, num_iters = 100):
        self.initialize_variational_dpmixture()
        phi = self.initialize_variational_data_assignment(len(X))

        for i in tqdm.tqmd(num_iters):
            # E-step: compute expectations
            component_resp = self.expected_component_resp()
            data_resp = self.obs_dist.expected_data_resp(X)

            # M-step: update variation paramters
            phi = self.update_variational_data_assignments(component_resp, data_resp)
            self.update_variational_dpmixture(phi)
            self.obs_dist.update(phi,X)

    def likelihood_allocation(self, X, phi):

        phi = phi.copy()

        q_zn_gt_i = self.compute_zn_gt_i(phi)
        E_log_V, E_log_1_min_V = self.compute_stick_log_expectations()

        term1 = q_zn_gt_i
        term1[:,:-1] *= E_log_1_min_V.reshape(1,-1)

        term2 = phi
        term2[:,:-1] *= E_log_V.reshape(1,-1)

        Eq_log_P_Zn_V = (term1 + term2)
        Eq_log_P_Zn_V[:,-1] = 0.

        Eq_log_P_xn_Zn = np.log(phi)

        return Eq_log_P_Zn_V.sum() + Eq_log_P_xn_Zn.sum()


    def likelihood_sticks(self):
        pass


    


class MultinomialObs:
    def __init__(self, num_categories=None, num_components=None, alpha = 5.):
        self.num_categories = num_categories
        self.num_components = num_components
        self.alpha = alpha
        self.weights = np.random.dirichlet(alpha*np.ones(num_categories), size = num_components)

    def expected_data_resp(self, X):
        E_log_w = digamma(self.weights) - digamma(self.weights.sum(1)).reshape(1,-1)
        resp = X.dot(E_log_w)
        return resp

    def update(self, phi, X):
        # X: N x C, phi N x T
        self.weights = self.alpha + phi.T @ X #T x C


    def likelihood(self, phi, X):

        E_log_pxn_Zn = self.phi.T @ self.expected_data_resp(X)

        return E_log_pxn_Zn.sum()





    


        













    




    # def initialize_variational_observation(self):


    






    