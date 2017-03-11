"""
author arvidzt
learning bayes
"""
import numpy as np

class MultinomialNB(object):

    def __init__(self,alpha=1.0,class_prior=None):
        self.alpha = alpha
        self.class_prior = class_prior
        self.classes = None
        self.conditional_prob = None

    def _calculate_feature_prob(self,feature):
        values = np.unique(feature)
        total_num = float(len(feature))
        value_prob = {}
        for v in values:
            value_prob[v]=(np.sum(np.equal(feature,v))+self.alpha)/(total_num+len(values)*self.alpha)
        return value_prob

    def fit(self,X,y):
        self.classes = np.unique(y)
        class_num = len(self.classes)

        #prior prob
        self.class_prior = []
        sample_num = float(len(y))
        for c in self.classes:
            c_num = np.sum(np.equal(y,c))
            self.class_prior.append((c_num+self.alpha)/(sample_num+class_num*self.alpha))

        #conditional_prob

        self.conditional_prob = {}
        for c in self.classes:
            self.conditional_prob[c]={}
            for i in range(len(X[0])):
                feature = X[np.equal(y,c)][:,i]
                self.conditional_prob[c][i] = self._calculate_feature_prob(feature)
        return self

    def _predict_single_sample(self,x):
        label = -1
        max_prob = 0

        for c in range(len(self.classes)):
            cur_class_prob = self.class_prior[c]
            cur_conditional_prob = 1.0
            feature_prob = self.conditional_prob[self.classes[c]]
            j=0
            for feature_i in feature_prob.keys():
                cur_conditional_prob *= feature_prob[feature_i][x[j]]
                j+=1
            #prior plus conditional
            if cur_conditional_prob*cur_class_prob > max_prob:
                max_prob = cur_conditional_prob*cur_class_prob
                label = self.classes[c]
        return label

    def predict(self,X):
        if X.ndim == 1:
                return self._predict_single_sample(X)
        else:
            labels = []
            for i in range(X.shape[0]):
                labels.append(self._predict_single_sample(X[i]))
            return labels


import numpy as np
X = np.array([
                      [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],
                      [4,5,5,4,4,4,5,5,6,6,6,5,5,6,6]
             ])
X = X.T
y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

nb = MultinomialNB(alpha=1.0)
nb.fit(X,y)
print nb.predict(np.array([1,4]))
