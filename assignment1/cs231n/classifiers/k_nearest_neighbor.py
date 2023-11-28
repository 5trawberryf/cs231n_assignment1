from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        print(X.shape)
        print(self.X_train.shape)
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
               a = X[i]-self.X_train[j]
               a = a*a
               dists[i][j]=np.sum(a)
                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        train = self.X_train.reshape(num_train,-1)
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            a = X[i] - train
            a = a * a
            dists[i] = np.sum(a,1)
           

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #train = self.X_train.reshape(1,num_train,-1)
        #train_multi = self.X_train.reshape(1,num_train,-1,1)
        #test = X.reshape(num_test,1,-1)
        #test_multi = X.reshape(num_test,1,1,-1)
        #a = np.square(train - test)
        #dists = np.sqrt(np.sum(a,2))
        #train_squ_sum = np.sum(np.square(train),2)
        #test_squ_sum = np.sum(np.square(test),2)
        #train_test_multi = np.matmul(test_multi,train_multi)
        #dists = np.sqrt(train_squ_sum + test_squ_sum - 2 * train_test_multi)
        train_trans = self.X_train.transpose()
        test_train_multi = X.dot(train_trans)
        train_squ_sum = np.sum(np.square(train_trans),0).reshape(1,-1)
        print(train_trans.shape)
        test_squ_sum = np.sum(np.square(X),1).reshape(-1,1)
        dists = train_squ_sum + test_squ_sum 
        dists = dists - 2 * test_train_multi
        print(train_squ_sum.shape)
        print(test_squ_sum.shape)
        print(test_train_multi.shape)
        print(dists.shape)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # y = self.X_train.transpose()
        # temp = X.dot(y)
        # temp  = temp * 2
        # print(X.shape)
        # print(y.shape)
        # x2=np.sum(np.square(X),axis=1).reshape(-1,1)
        # y2=np.sum(np.square(y),axis=0).reshape(1,-1)
        # print(temp.shape)
        # print(x2.shape)
        # print(y2.shape)
        # c=x2+y2
        # print(c.shape)
        # temp=c-temp      
        # print(temp.shape)
        # dists=temp
        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        num_train = dists.shape[1]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            ##########################################
            # ###############################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #my code:
            #index_array = np.argsort(dists[i],1)[:,:k]
            #closest_y = self.y_train[np.argsort(dists[i],1)[:,:k]]
            #the right code
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #values,num_uni = np.unique(closest_y,return_counts=True,axis=1)
            #indices = np.argmax(num_uni, axis=1)
            #y_pred = np.take_along_axis(values, indices[:, None], axis=1)
            counts = np.bincount(closest_y)
            y_pred[i] = np.argmax(counts)

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred
