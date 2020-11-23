import numpy as np

'''
Naive Bayes classifier
'''

class NBmodel:
    def __init__(self, phi_y = 0, phi_jy = None):
        self.phi_y = phi_y
        self.phi_jy = phi_jy

    def fit(self, X, labels):
        """  
        Args:
            X: A numpy matrix containing words counts is in the training data
            labels: The binary (0 or 1) labels for that training data
        Returns: 
        """
        n, m = X.shape
        self.phi_y = np.mean(labels)
        self.phi_jy = np.zeros((2, m))

        d = np.sum(X, axis=1)

        self.phi_jy[0, :] = (1+np.sum(X[labels==0, :], axis = 0))/ (sum(d[labels==0])+n)
        self.phi_jy[1, :] = (1+np.sum(X[labels==1, :], axis = 0))/ (sum(d[labels==1])+n)

    def predict(self, X):
        """  
        Args:
            X: A numpy matrix containing word counts is in the dev/test data
        Returns: 
            - binary prediction (0 or 1) labels for that dev/test data
        """
        nom = np.prod(np.power(self.phi_jy[1, :], X).T, axis = 0) * self.phi_y
        denom = nom + np.prod(np.power(self.phi_jy[0, :], X).T, axis = 0) * (1-self.phi_y)
        probs = nom / denom
        return probs > 0.5
    
    def get_topN(self, dictionary, N = 5):
        """Compute the top five words that are most indicative of the spam (i.e positive) class.

        Args:
            dictionary: A mapping of word to integer ids

        Returns: A list of the top five most indicative words in sorted order with the most indicative first
        """
        scores = np.log(self.phi_jy[1, :]/ self.phi_jy[0, :])
        return [sorted(dictionary.keys())[i] for i in np.argsort(scores)[-N:]]