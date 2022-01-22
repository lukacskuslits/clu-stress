import pandas as pd
import matplotlib.pyplot as plt
from codes.clu_stress import CluStress
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs

## Synthetic blobs
# -------------------------------------------------------------------------------------------- #

pontok = make_blobs(n_samples=100, n_features=2, centers=3, random_state=0, cluster_std=0.4)
scaler = MinMaxScaler()
pontok_x = scaler.fit_transform(pontok[0][:,0].reshape(-1, 1)).flatten()
pontok_y = scaler.fit_transform(pontok[0][:,1].reshape(-1, 1)).flatten()
data = pd.DataFrame({'x': list(pontok_x), 'y': list(pontok_y)})
data['cluster_fl'] = 0
data['PID'] = data.index
#data = pd.read_csv('./test_data.csv')
#print(data.head())
variables = data.columns[:-2]
cols = data.columns
data = data[cols]
print(variables)
print(cols)

clu_stress_obj = CluStress(data)
print(data)
[data, data_outlier, clu_stress_obj.dist] = clu_stress_obj.select_outliers(data)
print(data_outlier)
pontok = clu_stress_obj.commence_clustering(data, variables)

n_clusters_h = len(pontok.cluster_fl.unique())
if n_clusters_h == -1:
    n_clusters_h = 'noise'
ax = plt.subplot(1, 1, 1)
plt.scatter(pontok['x'], pontok['y'], c=pontok.cluster_fl, cmap='rainbow')
ax.set_aspect(1)
plt.title('CluStress automated with ' + str(n_clusters_h) + ' clusters')
plt.rcParams.update({'font.size': 15})
plt.savefig('CluStress_Blobs.png')
plt.show()

## Vrancea earthquake FMS
# -------------------------------------------------------------------------------------------- #


