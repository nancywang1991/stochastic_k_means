import random
import numpy as np
import stochastic_k
import pdb
import time
import cPickle as pickle

def gauss_data(n_clusters, n_data_points, dist, stdev, dim):
    data =[]
    for c in xrange(n_clusters):
        data.append(np.hstack([np.random.multivariate_normal(np.zeros(dim)+dist*c, np.identity(dim)*stdev, size=n_data_points),
                  np.zeros(shape=(n_data_points,1))+c]))
    return np.vstack(data)

def cluster_accuracy(true, n_cat, clustered, n_clusters):
    accuracy_l = np.zeros(n_cat)
    accuracy_loc = np.zeros(n_cat)
    for c in xrange(n_cat):
        actual_cat=np.where(true==c)[0]
        accuracies = [len((set(actual_cat).intersection(np.where(clustered==d)[0]))) for d in xrange(n_clusters)]
        accuracy_l[c] = np.max(accuracies)/float(len(actual_cat))
        accuracy_loc[c] = np.argmax(accuracies)
    return accuracy_l, accuracy_loc

def cluster_procedure(n_jobs, n_cluster, n_data, dist, stdev, n_dim):
     result=np.zeros(shape=(2,3))
     print str(n_jobs) + "_" + str(n_cluster) + "_" + str(n_data) + "_" + str(dist) + "_" + str(stdev) + "_" + str(n_dim)
     data = gauss_data(n_cluster, n_data, dist, stdev, n_dim)
     #Stochastic K-means
     result_temp=[]
     time_temp=[]
     invalid_cnt=0
     for x in xrange(20):
         start = time.time()
         stoch_k = stochastic_k.stochastic_k(n_clusters=n_cluster, max_iter=50, n_jobs=n_jobs)
         centers, labels, inertia = stoch_k.fit(data[:,:-1])
         time_temp.append(time.time()-start)
         accuracy, accuracy_loc = cluster_accuracy(data[:,-1], n_cluster, labels, n_cluster)
         result_temp.append(np.mean(accuracy))
         if len(set(accuracy_loc))<len(accuracy_loc):
             invalid_cnt += 1
     result[0,0]=np.mean(time_temp)
     result[0,1]=np.mean(result_temp)
     result[0,2]=invalid_cnt

     #Normal K-means
     if n_jobs==1:
         result_temp=[]
         time_temp=[]
         invalid_cnt=0
         for x in xrange(20):
             start = time.time()
             normal_k = stochastic_k.stochastic_k(n_clusters=n_cluster)
             centers, labels, inertia = normal_k.fit(data[:,:-1])
             time_temp.append(time.time()-start)
             accuracy, accuracy_loc = cluster_accuracy(data[:,-1], n_cluster, labels, n_cluster)
             result_temp.append(np.mean(accuracy))
             if len(set(accuracy_loc))<len(accuracy_loc):
                 invalid_cnt += 1
         result[1,0]=np.mean(time_temp)
         result[1,1]=np.mean(result_temp)
         result[1,2]=invalid_cnt
     print result
     return result

def main():
    default_params_l=[[1,3,1000,1,0.5,10], [4,3,10000,1,0.5,10]]
    n_jobs_range=range(1,7)
    n_cluster_range=range(2,8)
    n_data_range=range(1000,10000,1000)
    stdev_range=list(np.array(range(1,20,1))/10.0)
    n_dim_range=range(1, 40, 5)
    for n, default_params in enumerate(default_params_l):
        jobs_res=np.zeros(shape=(len(n_jobs_range),2,3))
        cluster_res=np.zeros(shape=(len(n_cluster_range),2,3))
        data_res=np.zeros(shape=(len(n_data_range),2,3))
        stdev_res=np.zeros(shape=(len(stdev_range),2,3))
        dim_res=np.zeros(shape=(len(n_dim_range),2,3))
        for a, n_jobs in enumerate(n_jobs_range):
            jobs_res[a,:,:]=cluster_procedure(n_jobs, default_params[1], default_params[2], default_params[3], default_params[4], default_params[5])
            pickle.dump(jobs_res, open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\jobs_result_default' + str(n) + '.p', 'wb'))
        for b, n_cluster in enumerate(n_cluster_range):
            cluster_res[b,:,:]=cluster_procedure(default_params[0], n_cluster, default_params[2], default_params[3], default_params[4], default_params[5])
            pickle.dump(cluster_res, open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\cluster_result_default' + str(n) + '.p', 'wb'))
        for c, n_data in enumerate(n_data_range):
            data_res[c,:,:]=cluster_procedure(default_params[0], default_params[1], n_data, default_params[3], default_params[4], default_params[5])
            pickle.dump(data_res, open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\data_result_default' + str(n) + '.p', 'wb'))
        for d, stdev in enumerate(stdev_range):
            stdev_res[d,:,:]=cluster_procedure(default_params[0], default_params[1], default_params[2], default_params[3], stdev, default_params[5])
            pickle.dump(stdev_res, open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\stdev_result_default' + str(n) + '.p', 'wb'))
        for e, n_dim in enumerate(n_dim_range):
            dim_res[e,:,:]=cluster_procedure(default_params[0], default_params[1], default_params[2], default_params[3], default_params[4], n_dim)
            pickle.dump(dim_res, open('C:\\Users\\wangnxr\\Documents\\classes\\systems\\project_data\\dim_result_default' + str(n) + '.p', 'wb'))

if __name__ == '__main__':
    main()