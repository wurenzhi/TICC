import numpy as np
from sklearn import mixture,metrics
import collections
from cvxpy import *
import random
from tgraphvx import TGraphVX


class TICC(object):
    def __init__(self,window_size = 10,num_clusters = 5, lam_sparse = 11e-2, switch_penalty = 400, maxIters = 1000,seed = 102):
        self.window_size = window_size
        self.num_clusters = num_clusters
        self.lam_sparse = lam_sparse
        self.switch_penalty = switch_penalty
        self.maxIters = maxIters
        self.train_cluster_inverse = {}
        self.cluster_mean_stacked_info = {}
        self.computed_covariance = {}
        np.random.seed(seed)

    def fit(self, xtrain, ytrain=None):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.complete_D_train = self.stacked_xtrain()
        self.clustered_points = self.initialize_cluster_with_gmm()
        self.optimize()
        return self.clustered_points, self.train_cluster_inverse

    def get_score(self):
        if self.ytrain is None:
            raise Exception('No label specified')
        return metrics.adjusted_rand_score(self.ytrain[:-self.window_size+1], self.clustered_points)



    def stacked_xtrain(self):
        (m, n) = self.xtrain.shape
        complete_D_train = np.zeros([m-self.window_size+1, self.window_size * n])
        for i in xrange(m-self.window_size+1):
            for k in xrange(self.window_size):
                if i + k < m-self.window_size+1:
                    complete_D_train[i][k * n:(k + 1) * n] = self.xtrain[i + k]
        return complete_D_train

    def initialize_cluster_with_gmm(self):
        gmm = mixture.GaussianMixture(n_components=self.num_clusters, covariance_type="full")
        gmm.fit(self.complete_D_train)
        clustered_points = gmm.predict(self.complete_D_train)
        return clustered_points

    def point_cluster_to_cluster_points(self,clustered_points):
        train_clusters = collections.defaultdict(list)
        for point in range(len(clustered_points)):
            cluster = clustered_points[point]
            train_clusters[cluster].append(point)
        return train_clusters

    def optimize(self):
#        log_det_values = {}
#        cluster_mean_info = {}
        m,n = self.xtrain.shape
        for iters in xrange(self.maxIters):
            print "\n\n\nITERATION ###", iters
            old_clustered_points = self.clustered_points
            if self.ytrain is not None:
                print "score: ", self.get_score()
            train_clusters = self.point_cluster_to_cluster_points(self.clustered_points)
            ##train_clusters holds the indices in complete_D_train
            ##for each of the clusters
            for cluster in xrange(self.num_clusters):
                if train_clusters[cluster]:
                    ##extract rows belong to current cluster from complete_D_train
                    indices = train_clusters[cluster]
                    D_train = np.zeros([len(train_clusters[cluster]), self.window_size * n])
                    for i in xrange(len(train_clusters[cluster])):
                        point = indices[i]
                        D_train[i, :] = self.complete_D_train[point, :]

                    ## solve toeplitz graphical lasso
                    val = self.solve_toeplitz_lasso(D_train)

                    S_est = self.upper2Full(val, 0)
                    u, _ = np.linalg.eig(S_est)
                    cov_out = np.linalg.inv(S_est)
                    ##Store the log-det, covariance, inverse-covariance, cluster means, stacked means
#                    log_det_values[self.num_clusters, cluster] = np.log(np.linalg.det(cov_out))
                    self.computed_covariance[self.num_clusters, cluster] = cov_out
#                    cluster_mean_info[self.num_clusters, cluster] = np.mean(D_train, axis=0)[
#                                                               (self.window_size - 1) * n:self.window_size * n].reshape([1, n])

                    self.cluster_mean_stacked_info[self.num_clusters, cluster] = np.mean(D_train, axis=0)
                    self.train_cluster_inverse[cluster] = S_est

                    print "OPTIMIZATION for Cluster #", cluster, "DONE!!!"

            cluster_norms = list(np.zeros(self.num_clusters))
            #        for cluster in xrange(num_clusters):
            #            print "length of the cluster ", cluster, "------>", len_train_clusters[cluster]

            ##Computing the norms
            if iters != 0:
                for cluster in xrange(self.num_clusters):
                    cluster_norms[cluster] = (np.linalg.norm(old_computed_covariance[self.num_clusters, cluster]), cluster)
                sorted_cluster_norms = sorted(cluster_norms, reverse=True)
            for cluster in xrange(self.num_clusters):
                if len(train_clusters[cluster]) == 0:
                    self.deal_with_empty_clusters(cluster, train_clusters, sorted_cluster_norms, old_computed_covariance)
            old_computed_covariance = self.computed_covariance
            print "UPDATED THE OLD COVARIANCE"

            ##Update cluster points(cluster assignment) - using NEW smoothening
            self.smoothening()
            for cluster in xrange(self.num_clusters):
                print "length of cluster #", cluster, "-------->", sum([x == cluster for x in self.clustered_points])

            if np.array_equal(old_clustered_points, self.clustered_points):
                print "\n\n\n\nCONVERGED!!! BREAKING EARLY!!!"
                break

    def deal_with_empty_clusters(self,cluster, train_clusters, sorted_cluster_norms, old_computed_covariance):
        ##Add a point to the empty clusters
        ##Assumption more non empty clusters than empty ones
        ##Add a point to the cluster
        while len(train_clusters[sorted_cluster_norms[counter][1]]) == 0:
            print "counter is:", counter
            counter += 1
            counter = counter % self.num_clusters
            print "counter is:", counter

        cluster_selected = sorted_cluster_norms[counter][1]
        print "cluster that is zero is:", cluster, "selected cluster instead is:", cluster_selected
        break_flag = False
        while not break_flag:
            point_num = random.randint(0, len(self.clustered_points))
            if self.clustered_points[point_num] == cluster_selected:
                self.clustered_points[point_num] = cluster
                # print "old covariances shape", old_computed_covariance[cluster_selected].shape
                self.computed_covariance[self.num_clusters, cluster] = old_computed_covariance[
                    self.num_clusters, cluster_selected]
                self.cluster_mean_stacked_info[self.num_clusters, cluster] = self.complete_D_train[point_num, :]
                #                            self.cluster_mean_info[self.num_clusters, cluster] = self.complete_D_train[point, :][
                #                                                                       (self.window_size - 1) * n:self.window_size * n]
                break_flag = True

    def upper2Full(self, a, eps=0):
        ind = (a < eps) & (a > -eps)
        a[ind] = 0
        n = int((-1 + np.sqrt(1 + 8 * a.shape[0])) / 2)
        A = np.zeros([n, n])
        A[np.triu_indices(n)] = a
        temp = A.diagonal()
        A = np.asarray((A + A.T) - np.diag(temp))
        return A

    def updateClusters(self, LLE_node_vals):
        """
        Takes in LLE_node_vals matrix and computes the path that minimizes
        the total cost over the path
        Note the LLE's are negative of the true LLE's actually!!!!!

        Note: switch penalty > 0
        """
        (T, num_clusters) = LLE_node_vals.shape
        future_cost_vals = np.zeros(LLE_node_vals.shape)

        ##compute future costs
        for i in xrange(T - 2, -1, -1):
            j = i + 1
            indicator = np.zeros(num_clusters)
            future_costs = future_cost_vals[j, :]
            lle_vals = LLE_node_vals[j, :]
            for cluster in xrange(num_clusters):
                total_vals = future_costs + lle_vals + self.switch_penalty
                total_vals[cluster] -= self.switch_penalty
                future_cost_vals[i, cluster] = np.min(total_vals)

        ##compute the best path
        path = np.zeros(T)

        ##the first location
        curr_location = np.argmin(future_cost_vals[0, :] + LLE_node_vals[0, :])
        path[0] = curr_location

        ##compute the path
        for i in xrange(T - 1):
            j = i + 1
            future_costs = future_cost_vals[j, :]
            lle_vals = LLE_node_vals[j, :]
            total_vals = future_costs + lle_vals + self.switch_penalty
            total_vals[int(path[i])] -= self.switch_penalty

            path[i + 1] = np.argmin(total_vals)

        ##return the computed path
        return path

    def solve_toeplitz_lasso(self, D_train):
        ##Fit a model - OPTIMIZATION
        size_blocks = self.xtrain.shape[1]
        probSize = self.window_size * size_blocks
        lamb = np.zeros((probSize, probSize)) + self.lam_sparse
        S = np.cov(np.transpose(D_train))

        # COPY THIS CODE
        gvx = TGraphVX()
        theta = semidefinite(probSize, name='theta')
        obj = -log_det(theta) + trace(S * theta)
        gvx.AddNode(0, obj)
        gvx.AddNode(1)
        dummy = Variable(1)
        gvx.AddEdge(0, 1, Objective=lamb * dummy + self.window_size * dummy + size_blocks * dummy)
        gvx.Solve(Verbose=False, MaxIters=1000, Rho=1, EpsAbs=1e-6, EpsRel=1e-6)
        return gvx.GetNodeValue(0, 'theta')

    def smoothening(self):
        ##Code -----------------------SMOOTHENING
        ##For each point compute the LLE
        print "beginning with the smoothening ALGORITHM"
        m,n = self.xtrain.shape
        inv_cov_dict = {}
        log_det_dict = {}
        for cluster in xrange(self.num_clusters):
            cov_matrix = self.computed_covariance[self.num_clusters, cluster][0:self.window_size * n, 0:self.window_size* n]
            inv_cov_matrix = np.linalg.inv(cov_matrix)
            log_det_cov = np.log(np.linalg.det(cov_matrix))  # log(det(sigma2|1))
            inv_cov_dict[cluster] = inv_cov_matrix
            log_det_dict[cluster] = log_det_cov

        LLE_all_points_clusters = np.zeros([len(self.clustered_points), self.num_clusters])
        for point in xrange(len(self.clustered_points)):
            # print "Point #", point
            if point + self.window_size - 1 < self.complete_D_train.shape[0]:
                for cluster in xrange(self.num_clusters):
                    cluster_mean_stacked = self.cluster_mean_stacked_info[self.num_clusters, cluster]
                    x = self.complete_D_train[point, :] - cluster_mean_stacked[0:(self.window_size) * n]
                    inv_cov_matrix = inv_cov_dict[cluster]  # np.linalg.inv(cov_matrix)
                    log_det_cov = log_det_dict[cluster]  # np.log(np.linalg.det(cov_matrix))# log(det(sigma2|1))
                    lle = np.dot(x.reshape([1, (self.window_size) * n]),
                                 np.dot(inv_cov_matrix, x.reshape([n * (self.window_size), 1]))) + log_det_cov
                    LLE_all_points_clusters[point, cluster] = lle
        self.clustered_points = self.updateClusters(LLE_all_points_clusters)

ticc = TICC(window_size = 6,num_clusters = 3, lam_sparse = 11e-2, switch_penalty = 100, maxIters = 1000)
Data = np.loadtxt('data.csv', delimiter=",")
(cluster_assignment, cluster_MRFs) = ticc.fit(Data[:,:-1],Data[:,-1])

print ticc.get_score()
