import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from codes.clu_stress import CluStress
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from codes.earthquake_fms import set_rakes

variables = ['x', 'y', 'z'] #, 'r']
data = pd.read_csv('synthetic_fms_020421_labeled.csv')
data_true = data.copy()
data = data[variables]
data_x = data.x.copy()
data_y = data.y.copy()
if 'r' in variables:
    data = set_rakes(data)
data['PID'] = data_true.index.copy()
data_true['PID'] = data_true.index.copy()
data['cluster_fl'] = 0
scaler = MinMaxScaler()
data[variables] = scaler.fit_transform(data[variables])
if 'r' in variables:
    data.r = 1.5*data.r
clu_stress_obj = CluStress(data)
[data, data_outlier, clu_stress_obj.dist] = clu_stress_obj.select_outliers(data, k=20)
data = clu_stress_obj.commence_clustering(data, variables)
data = pd.concat([data, data_outlier])

data['x_raw'] = data_x
data['y_raw'] = data_y
# n_clusters_h = data.cluster_fl.max()
# if n_clusters_h == -1:
#     n_clusters_h = 'noise'
# ax = plt.subplot(1,1,1)
# scatter = plt.scatter(data['x_raw'],data['y_raw'], c=data.cluster_fl, cmap='rainbow')
# plt.xlabel('x')
# plt.ylabel('y')
# ax.set_xlim(xmin=0, xmax=300)
# plt.title('Automatized with ' + str(n_clusters_h) + ' clusters')
# plt.rcParams.update({'font.size': 15})
# legend = ax.legend(*scatter.legend_elements(),
#                    loc="upper right", title="Classes", prop={"size":10})
# ax.add_artist(legend)
# plt.savefig('CluStress_test.png')
# plt.show()

#
cluster_centres_x = []
cluster_centres_y = []
labels = []
for c_label in list(data_true.Cluster.unique()):
    cluster_centres_x.append(data_true[data_true.Cluster==c_label].x.mean())
    cluster_centres_y.append(data_true[data_true.Cluster==c_label].y.mean())
    labels.append(c_label)
cluster_centres_true = pd.DataFrame({'centre_x': cluster_centres_x, 'centre_y': cluster_centres_y, 'Cluster': labels})
print(cluster_centres_true)
#
#
cluster_centres_x = []
cluster_centres_y = []
labels = []
for c_label in list(data.cluster_fl.unique()):
    cluster_centres_x.append(data[data.cluster_fl==c_label].x_raw.mean())
    cluster_centres_y.append(data[data.cluster_fl==c_label].y_raw.mean())
    labels.append(c_label)
cluster_centres_pred = pd.DataFrame({'centre_x': cluster_centres_x, 'centre_y': cluster_centres_y, 'cluster_fl': labels})
print(cluster_centres_pred)
#
def dist_euclidean(x1, x2, y1, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

cluster_centres = cluster_centres_pred
cluster_centres['nearest_cluster_label'] = -2
cluster_centres['distance'] = 0
for cluster_flag in list(cluster_centres.cluster_fl.values):
    cluster_centres_part = cluster_centres[cluster_centres.cluster_fl==cluster_flag]
    distances = []
    flags = []
    for ind in list(cluster_centres_true.index):
        true_centre_x = cluster_centres_true.centre_x.iloc[ind]
        true_centre_y = cluster_centres_true.centre_y.iloc[ind]
        cluster = cluster_centres_true.Cluster.iloc[ind]
        distances.append(dist_euclidean(cluster_centres_part.centre_x, true_centre_x, cluster_centres_part.centre_y, true_centre_y))
        flags.append(cluster)
    dist_data = pd.DataFrame({'dist': distances, 'flag': flags})
    print(dist_data.flag.loc[dist_data.dist==dist_data.dist.min()].values[0])
    label_value = dist_data.flag.loc[dist_data.dist==dist_data.dist.min()].values[0]
    cluster_centres.nearest_cluster_label.loc[cluster_centres.cluster_fl==cluster_flag] = label_value
    cluster_centres.distance.loc[cluster_centres.cluster_fl==cluster_flag] = dist_data.dist.min()

for true_cluster_label in list(cluster_centres_true.Cluster.values):
    cluster_centres_partition = cluster_centres[cluster_centres.nearest_cluster_label==true_cluster_label]
    cond_1 = cluster_centres.nearest_cluster_label==true_cluster_label
    cond_2 = cluster_centres.distance!=cluster_centres_partition.distance.min()
    cluster_centres.nearest_cluster_label.loc[cond_1 & cond_2] = -2
cluster_centres.drop(columns=['distance'])
print(cluster_centres)
data['new_cluster_fl'] = -2
for cluster_flag in list(cluster_centres.cluster_fl.values):
    nearest_label = cluster_centres[cluster_centres.cluster_fl==cluster_flag].nearest_cluster_label.values[0]
    if nearest_label != -2:
        data.new_cluster_fl.loc[data.cluster_fl==cluster_flag] = nearest_label

data.new_cluster_fl.loc[data.new_cluster_fl==-2] = data.new_cluster_fl.max() + 1
print(data.new_cluster_fl.unique())

n_clusters_h = data.new_cluster_fl.max()
if n_clusters_h == -1:
    n_clusters_h = 'noise'
ax = plt.subplot(1,1,1)
scatter = plt.scatter(data['x_raw'],data['y_raw'], c=data.new_cluster_fl, cmap='rainbow')
plt.xlabel('x')
plt.ylabel('y')
ax.set_xlim(xmin=0, xmax=300)
plt.title('Automatized with ' + str(n_clusters_h) + ' clusters')
plt.rcParams.update({'font.size': 15})
legend = ax.legend(*scatter.legend_elements(),
                   loc="upper right", title="Classes", prop={"size":10})
ax.add_artist(legend)
plt.savefig('CluStress_test.png')
plt.show()

data = data.sort_values(by='PID')
data_true = data_true.sort_values(by='PID')
cm = confusion_matrix(data_true['Cluster'], data['new_cluster_fl'])
print('Confusion Matrix : \n', cm)

total=sum(sum(cm))
accuracy=(np.trace(cm))/total
print('Accuracy : ', accuracy)

categorical_accuracies = []
for ii in range(cm.shape[0]):
    categorical_accuracies.append(cm[ii,ii]/(sum(cm[ii,:])+sum(cm[:,ii])-cm[ii,ii]))

print(categorical_accuracies)

