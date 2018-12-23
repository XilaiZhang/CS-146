"""
Author      : Yi-Chieh Wu, Sriram Sankararaman
Description : Titanic
"""

# Use only the provided packages!
#import matplotlib
#matplotlib.use('Agg')
import math
import csv
from util import *
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics

######################################################################
# classes
######################################################################

class Classifier(object) :
    """
    Classifier interface.
    """
    
    def fit(self, X, y):
        raise NotImplementedError()
        
    def predict(self, X):
        raise NotImplementedError()


class MajorityVoteClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that always predicts the majority class.
        
        Attributes
        --------------------
            prediction_ -- majority class
        """
        self.prediction_ = None
    
    def fit(self, X, y) :
        """
        Build a majority vote classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        majority_val = Counter(y).most_common(1)[0][0]
        self.prediction_ = majority_val
        return self
    
    def predict(self, X) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.prediction_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        
        n,d = X.shape
        y = [self.prediction_] * n 
        return y


class RandomClassifier(Classifier) :
    
    def __init__(self) :
        """
        A classifier that predicts according to the distribution of the classes.
        
        Attributes
        --------------------
            probabilities_ -- class distribution dict (key = class, val = probability of class)
        """
        self.probabilities_ = None
    
    def fit(self, X, y) :
        """
        Build a random classifier from the training set (X, y).
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            y    -- numpy array of shape (n,), target classes
        
        Returns
        --------------------
            self -- an instance of self
        """
        
        ### ========== TODO : START ========== ###
        # part b: set self.probabilities_ according to the training set
        percentone= Counter(y).most_common(2)[0][1]/ (Counter(y).most_common(2)[0][1] + Counter(y).most_common(2)[1][1])
        if Counter(y).most_common(2)[0][0] == 0:
            percentone= 1-percentone
        self.probabilities_ = percentone
        ### ========== TODO : END ========== ###
        
        return self
    
    def predict(self, X, seed=1234) :
        """
        Predict class values.
        
        Parameters
        --------------------
            X    -- numpy array of shape (n,d), samples
            seed -- integer, random seed
        
        Returns
        --------------------
            y    -- numpy array of shape (n,), predicted classes
        """
        if self.probabilities_ is None :
            raise Exception("Classifier not initialized. Perform a fit first.")
        np.random.seed(seed)
        
        ### ========== TODO : START ========== ###
        # part b: predict the class for each test example
        # hint: use np.random.choice (be careful of the parameters)
        
        y = np.random.choice(2,X.shape[0],p=[1-self.probabilities_,self.probabilities_])
        
        ### ========== TODO : END ========== ###
        
        return y


######################################################################
# functions
######################################################################
def plot_histograms(X, y, Xnames, yname) :
    n,d = X.shape  # n = number of examples, d =  number of features
    fig = plt.figure(figsize=(20,15))
    nrow = 3; ncol = 3
    for i in range(d) :
        fig.add_subplot (3,3,i)  
        data, bins, align, labels = plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname, show = False)
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xnames[i])
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
 
    plt.savefig ('histograms.pdf')


def plot_histogram(X, y, Xname, yname, show = True) :
    """
    Plots histogram of values in X grouped by y.
    
    Parameters
    --------------------
        X     -- numpy array of shape (n,d), feature values
        y     -- numpy array of shape (n,), target classes
        Xname -- string, name of feature
        yname -- string, name of target
    """
    
    # set up data for plotting
    targets = sorted(set(y))
    data = []; labels = []
    for target in targets :
        features = [X[i] for i in range(len(y)) if y[i] == target]
        data.append(features)
        labels.append('%s = %s' % (yname, target))
    
    # set up histogram bins
    features = set(X)
    nfeatures = len(features)
    test_range = list(range(int(math.floor(min(features))), int(math.ceil(max(features)))+1))
    if nfeatures < 10 and sorted(features) == test_range:
        bins = test_range + [test_range[-1] + 1] # add last bin
        align = 'left'
    else :
        bins = 10
        align = 'mid'
    
    # plot
    if show == True:
        plt.figure()
        n, bins, patches = plt.hist(data, bins=bins, align=align, alpha=0.5, label=labels)
        plt.xlabel(Xname)
        plt.ylabel('Frequency')
        plt.legend() #plt.legend(loc='upper left')
        plt.show()

    return data, bins, align, labels


def error(clf, X, y, ntrials=100, test_size=0.2) :
    """
    Computes the classifier error over a random split of the data,
    averaged over ntrials runs.
    
    Parameters
    --------------------
        clf         -- classifier
        X           -- numpy array of shape (n,d), features values
        y           -- numpy array of shape (n,), target classes
        ntrials     -- integer, number of trials
    
    Returns
    --------------------
        train_error -- float, training error
        test_error  -- float, test error
    """
    
    ### ========== TODO : START ========== ###
    # compute cross-validation error over ntrials
    # hint: use train_test_split (be careful of the parameters)
    
    train_error = 0
    test_error = 0    
    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        clf.fit(X_train,y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        temp_train_error = 1 - metrics.accuracy_score(y_train, train_pred, normalize=True) 
        temp_test_error = 1 - metrics.accuracy_score(y_test, test_pred, normalize=True)
        train_error += temp_train_error
        test_error += temp_test_error
    train_error/=ntrials
    test_error/=ntrials
    ### ========== TODO : END ========== ###
    
    return train_error, test_error


def write_predictions(y_pred, filename, yname=None) :
    """Write out predictions to csv file."""
    out = open(filename, 'wb')
    f = csv.writer(out)
    if yname :
        f.writerow([yname])
    f.writerows(list(zip(y_pred)))
    out.close()


######################################################################
# main
######################################################################

def main():
    # load Titanic dataset
    titanic = load_data("titanic_train.csv", header=1, predict_col=0)
    X = titanic.X; Xnames = titanic.Xnames
    y = titanic.y; yname = titanic.yname
    n,d = X.shape  # n = number of examples, d =  number of features
    
    
    
    #========================================
    # part a: plot histograms of each feature
    print('Plotting...')
    '''
    for i in range(d) :
        plot_histogram(X[:,i], y, Xname=Xnames[i], yname=yname)
    '''
       
    #========================================
    # train Majority Vote classifier on data
    print('Classifying using Majority Vote...')
    clf = MajorityVoteClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    
    
    
    ### ========== TODO : START ========== ###
    # part b: evaluate training error of Random classifier
    print('Classifying using Random...')
    clf = RandomClassifier() # create MajorityVote classifier, which includes all model parameters
    clf.fit(X, y)                  # fit training data using the classifier
    y_pred = clf.predict(X)        # take the classifier and run it on the training data
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part c: evaluate training error of Decision Tree classifier
    # use criterion of "entropy" for Information gain 
    print('Classifying using Decision Tree...')
    clf = DecisionTreeClassifier(criterion="entropy") 
    clf.fit(X, y)                  
    y_pred = clf.predict(X)       
    train_error = 1 - metrics.accuracy_score(y, y_pred, normalize=True)
    print('\t-- training error: %.3f' % train_error)
    ### ========== TODO : END ========== ###

    

    # note: uncomment out the following lines to output the Decision Tree graph
    """
    # save the classifier -- requires GraphViz and pydot
    import StringIO, pydot
    from sklearn import tree
    dot_data = StringIO.StringIO()
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=Xnames)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("dtree.pdf") 
    """



    ### ========== TODO : START ========== ###
    # part d: evaluate training error of k-Nearest Neighbors classifier
    # use k = 3, 5, 7 for n_neighbors 
    print('Classifying using k-Nearest Neighbors...')
    clf3 = KNeighborsClassifier(n_neighbors=3) 
    clf3.fit(X, y)                  
    y_pred3 = clf3.predict(X)       
    train_error3 = 1 - metrics.accuracy_score(y, y_pred3, normalize=True)
    print('\t-- training error: %.3f' % train_error3)

    clf5 = KNeighborsClassifier(n_neighbors=5) 
    clf5.fit(X, y)                  
    y_pred5 = clf5.predict(X)       
    train_error5 = 1 - metrics.accuracy_score(y, y_pred5, normalize=True)
    print('\t-- training error: %.3f' % train_error5)

    clf7 = KNeighborsClassifier(n_neighbors=7) 
    clf7.fit(X, y)                  
    y_pred7 = clf7.predict(X)       
    train_error7 = 1 - metrics.accuracy_score(y, y_pred7, normalize=True)
    print('\t-- training error: %.3f' % train_error7)
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part e: use cross-validation to compute average training and test error of classifiers
    print('Investigating various classifiers...')
    clf1=MajorityVoteClassifier()
    clf2=RandomClassifier()
    clf3=DecisionTreeClassifier(criterion="entropy")
    clf4=KNeighborsClassifier(n_neighbors=5)
    for clf in [clf1,clf2,clf3,clf4]:
        first, second= error(clf, X, y, ntrials=100, test_size=0.2)
        print('\t--%s :' %type(clf).__name__)
        print('\t-- training error: %.3f -- test error: %.3f' %(first, second) )
    ### ========== TODO : END ========== ###



    ### ========== TODO : START ========== ###
    # part f: use 10-fold cross-validation to find the best value of k for k-Nearest Neighbors classifier
    print('Finding the best k for KNeighbors classifier...')
    xdata=[]
    ydata=[]
    for i in range(1,50,2):
        scores = cross_val_score(KNeighborsClassifier(n_neighbors=i, p=2), X, y, cv=10)
        xdata.append(i)
        ydata.append(1-np.mean(scores))
    ymin=np.min(ydata)
    yindex=ydata.index(ymin)
    xindex=yindex
    '''
    plt.plot(xdata, ydata)
    plt.annotate('minimum={:d}'.format(xdata[xindex]), xy=(xdata[xindex], ymin), xytext=(xdata[xindex], 0.32),
            arrowprops=dict(facecolor='black', shrink=0.05),
                 )
   
    plt.xlabel('number of neighbors, k')
    plt.ylabel('validation error')
    plt.savefig( 'part f plot')
    '''
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part g: investigate decision tree classifier with various depths
    print('Investigating depths...')
    depth=[]
    train_error=[]
    test_error=[]
    for i in range(1,21):
        temp_train_error, temp_test_error = error(DecisionTreeClassifier(criterion='entropy', max_depth=i), X, y)
        depth.append(i)
        train_error.append(temp_train_error)
        test_error.append(temp_test_error)

    plt.plot(depth, train_error, 'b--',label="training error")
    plt.plot(depth, test_error, 'ys', label="test error")
    plt.xlabel('depth limit')
    plt.ylabel('training and test error rate')
    
    plt.legend(loc=3)

    plt.savefig('part g graph')
    ### ========== TODO : END ========== ###
    
    
    
    ### ========== TODO : START ========== ###
    # part h: investigate Decision Tree and k-Nearest Neighbors classifier with various training set sizes
    print('Investigating training set sizes...')
    percent= []
    decision_train_error= []
    decision_test_error= []
    knn_train_error= []
    knn_test_error= []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    for i in range(1,11):
        percent.append(i*0.1)
        X_train2, y_train2 = X_train, y_train
        if i!=10:
            X_train2, X_test_2, y_train2, y_test_2 = train_test_split(X_train, y_train, test_size=(1-i*0.1))
        clf1 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf2 = KNeighborsClassifier(n_neighbors=7)

        clf1.fit(X_train2,y_train2)
        predict_y_train=clf1.predict(X_train2)
        temp_train_error= 1-metrics.accuracy_score(y_train2, predict_y_train, normalize=True)
        decision_train_error.append(temp_train_error)
    
        predict_y_test=clf1.predict(X_test)
        temp_test_error_new=1 - metrics.accuracy_score(y_test, predict_y_test, normalize=True)
        decision_test_error.append(temp_test_error_new)

        clf2.fit(X_train2,y_train2)
        predict_y_train=clf2.predict(X_train2)
        temp_train_error= 1-metrics.accuracy_score(y_train2, predict_y_train, normalize=True)
        knn_train_error.append(temp_train_error)

        predict_y_test=clf2.predict(X_test)
        temp_test_error_new=1 - metrics.accuracy_score(y_test, predict_y_test, normalize=True)
        knn_test_error.append(temp_test_error_new)
    plt.clf()
    plt.plot(percent, decision_train_error, 'b--',label="decision tree training error")
    plt.plot(percent, decision_test_error, 'bs-', label="decision tree test error")
    plt.plot(percent, knn_train_error, 'g--',label="knn training error")
    plt.plot(percent, knn_test_error, 'gs-', label="knn test error")
    plt.xlabel('percent of the original training data')
    plt.ylabel('error rate')
    
    plt.legend(loc=4)

    plt.savefig('part h graph')
	### ========== TODO : END ========== ###
    
       
    print('Done')


if __name__ == "__main__":
    main()
