import math
import sys
import time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import iqr
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from clustress.distance_matrices_new import DistanceMatricesNew
import itertools
from itertools import compress
import matplotlib.pyplot as plt


class CluStress:

	def __init__(self, points):
		col_list = list(points.columns)
		print(col_list)
		if 'PID' not in col_list:
			print('Column PID missing from input data!')
			sys.exit(-1)
		if 'cluster_fl' not in col_list:
			print('Column cluster_fl missing from input data!')
			sys.exit(-1)
		dist_array = DistanceMatricesNew().compute_distance_matrix(points, col_list[:-2])
		self.dist_array = dist_array
		dist = DistanceMatricesNew().transform_distance_matrix(dist_array)
		dist['rev_id_col'] = dist['id_col'].apply(DistanceMatricesNew().flip_values)
		self.dist = dist

	def bin_distances(self):
		self.dist = self.dist.sort_values(by='distance')
		var_bins = np.histogram_bin_edges(self.dist['distance'].values, bins='auto')
		#Interpolate between bins (resampling)
		for step in range(1):
			n = len(var_bins)
			var_bin_range = np.arange(0, 2*n, 2)
			var_bin_range_new = np.arange(2*n-1)
			var_bins = np.interp(var_bin_range_new, var_bin_range, var_bins)
		print(len(var_bins))
		self.dist['distance'] = list(np.digitize(self.dist['distance'].values, var_bins))
		self.dist = self.dist.sort_values(by='id_0')

	def get_outlier_distances(self, k):
		"""Tukey's fence outlier detection."""
		nearest_dists = DistanceMatricesNew().find_nearest_dists(self.dist_array)
		quant_df = nearest_dists.distance.quantile([.25,.75])
		q_1 = quant_df.iloc[0]
		q_3 = quant_df.iloc[1]
		outlier_range = q_3 + k*(q_3 - q_1)
		return outlier_range

	def select_outliers(self, points, k=1.5):
		outlier_range = self.get_outlier_distances(k)
		dist_sorted = self.dist.sort_values(by=['id_0','distance']).copy()
		for _id in list(points['PID']):
			if _id == len(list(points['PID']))-1:
				nearest_dist_of_point = dist_sorted.loc[dist_sorted['id_1']==_id]['distance'].iloc[0]
			else:
				nearest_dist_of_point = dist_sorted.loc[dist_sorted['id_0']==_id]['distance'].iloc[0]
			if nearest_dist_of_point>outlier_range:
				points.cluster_fl.loc[points.PID==_id] = -1
		self.dist['cluster_fl'] = self.dist.id_0.map(points.set_index('PID')['cluster_fl'])
		self.dist['cluster_fl_1'] = self.dist.id_1.map(points.set_index('PID')['cluster_fl'])
		self.dist.cluster_fl.loc[self.dist.id_1 == points.PID.max()] = list(self.dist.cluster_fl_1.loc[self.dist.id_1 == points.PID.max()].astype(int))
		self.dist.drop(columns=['cluster_fl_1'])
		dist_filtered = self.dist.loc[self.dist['cluster_fl']!=-1].copy()
		dist_filtered = dist_filtered.drop(columns=['cluster_fl', 'cluster_fl_1'])
		points_filtered = points.loc[points['cluster_fl'] != -1].copy()
		points_outlier = points.loc[points['cluster_fl'] == -1].copy()
		return points_filtered, points_outlier, dist_filtered

	def transform_distance_matrix(self, distance_matrix):
		"""Converts distance_matrix to a sparse array dictionary of keys format."""
		dist_sparse = csr_matrix(np.triu(distance_matrix).copy())
		dist_dict = dist_sparse.todok()
		dist_dict = dict(dist_dict.items())
		dist_df = pd.DataFrame({'id_col': list(dist_dict.keys()), 'distance': list(dist_dict.values())})
		dist_df['id_0'] = dist_df['id_col'].apply(lambda x: x[0])
		dist_df['id_1'] = dist_df['id_col'].apply(lambda x: x[1])
		return dist_df

	def set_distance_matrix(self, clusterized_list):
		"""Keeps only those elements of the distance matrix which are not clusterized."""
		all_clusterized_pairs = list(itertools.combinations(clusterized_list, 2))
		self.dist = self.dist.loc[np.invert(self.dist.id_col.isin(all_clusterized_pairs))]
		self.dist = self.dist.loc[np.invert(self.dist.rev_id_col.isin(all_clusterized_pairs))]

	def find_connected_pairs(self, point_set, point_pairs):
		connected_pairs = ()
		for point in point_set:
			is_connected_pair = [point in i for i in point_pairs]
			connected_pairs += tuple(compress(point_pairs, is_connected_pair))
			point_pairs = list(compress(point_pairs, np.invert(is_connected_pair)))
			if len(point_pairs)==0:
				break
		return connected_pairs, point_pairs

	def connect_connected_pairs(self, point_set, connected_pairs):
		for connected_pair in connected_pairs:
			point_set += connected_pair
			point_set = tuple(set(point_set))
		return point_set

	def connect_nearest_points_to_pair(self, point_set, point_pairs):
		while True:
			point_set_old = point_set
			[connected_pairs, point_pairs] = self.find_connected_pairs(point_set, point_pairs)
			point_set = self.connect_connected_pairs(point_set, connected_pairs)
			if (len(point_set_old) == len(point_set)) or (len(point_pairs)==0):
				break
		return point_set, point_pairs

	def get_connected_points(self, nearest_point_pairs):
		remaining_point_pairs = nearest_point_pairs
		connected_point_pairs = []
		increment = 0
		L = len(nearest_point_pairs)
		while True:
			point_pair = nearest_point_pairs[increment]
			[connected_point_set, remaining_point_pairs] = self.connect_nearest_points_to_pair(point_pair, remaining_point_pairs)
			nearest_point_pairs = [connected_point_set] + remaining_point_pairs
			connected_point_pairs.append(connected_point_set)
			L=len(nearest_point_pairs)
			increment = increment+1
			if increment >= L:
				if len(remaining_point_pairs)!=0:
					[connected_point_set, _] = self.connect_nearest_points_to_pair(remaining_point_pairs[0], remaining_point_pairs)
					connected_point_pairs.append(connected_point_set)
				break
		return connected_point_pairs

	def clusterize_nearest(self, points, cluster_fl):
		if cluster_fl != 0:
			clusterized = points.loc[(points['cluster_fl'] != 0)].reset_index(drop=True) #(points['cluster_fl'] <= cluster_fl) &
		else:
			clusterized = points.loc[points['cluster_fl'] != cluster_fl].reset_index(drop=True)
		self.set_distance_matrix(list(clusterized.PID))
		clusterized_id = list(clusterized.PID)
		disttemp = self.dist.loc[~(self.dist.id_0.isin(clusterized_id)) & ~(self.dist.id_1.isin(clusterized_id))].copy()
		indexes_of_nearest_points = list(disttemp.loc[(disttemp.distance == disttemp.distance.min())].id_col)
		connected_nearest_points = self.get_connected_points(indexes_of_nearest_points)
		for connected_point_set in connected_nearest_points:
			cluster_fl = int(cluster_fl + 1)
			points['cluster_fl'][points.PID.isin(list(connected_point_set))] = int(cluster_fl)
			clusterized = points.loc[points['cluster_fl'] == cluster_fl].reset_index(drop=True)
			self.set_distance_matrix(list(clusterized.PID))
		points = points.reset_index(drop=True)
		return points, cluster_fl

	def check_if_nearest_point_is_part_of_a_cluster(self, clusterized, to_merge, variables):
		to_merge = self.join_with_clusterized(clusterized, to_merge, 'id_1', variables)
		to_merge['point_id'] = to_merge.index
		to_merge = to_merge[to_merge.cluster_fl > 0]
		return to_merge

	def join_with_clusterized(self, clusterized, to_merge, point_id_to_check, variables):
		to_merge = to_merge.set_index(point_id_to_check)
		to_merge = to_merge.drop(columns=['cluster_fl'], axis=1).join(
		clusterized.drop(columns=variables, axis=1).set_index('PID'), how='left')
		to_merge = to_merge.fillna(0)
		return to_merge

	def merge_points(self, points, to_merge):
		points['point_id'] = points['PID']
		if 0 in list(points.columns):
			points = points.drop(columns=[0], axis=1)
		if 0 in list(to_merge.columns):
			to_merge = to_merge.drop(columns=[0], axis=1)
		points = points.set_index('point_id').join(to_merge.set_index('id_0'), how='left')
		points = points.fillna(0)
		points['cluster_fl'] = points['cluster_fl'] + points['cluster_fl_1']
		points['cluster_fl'] = points['cluster_fl'].astype(int)
		points = points.drop(columns=['cluster_fl_1'], axis=1)
		return points

	def link_points_which_have_their_nearest_neighbours_in_a_cluster(self, points, variables):
		not_clusterized = points.loc[points['cluster_fl'] == 0]
		not_clusterized_id = list(not_clusterized.PID)
		point_id = list(points.PID.astype(int))
		clusterized = points.loc[points['cluster_fl'] != 0]
		not_clust_pairs = [(nc, p) for nc, p in itertools.product(not_clusterized_id, point_id)]
		id_0 = list(itertools.chain.from_iterable(itertools.repeat(x, len(point_id)) for x in not_clusterized_id))
		id_1 = point_id * len(not_clusterized_id)
		nearest_neighbour_table = pd.DataFrame({'id_col': not_clust_pairs, 'id_0': id_0, 'id_1': id_1})
		nearest_neighbour_table['distance'] = nearest_neighbour_table.id_col.map(self.dist.set_index('id_col')['distance'])
		nearest_neighbour_table['distance_0'] = nearest_neighbour_table['distance']
		nearest_neighbour_table = nearest_neighbour_table.drop(columns=['distance'], axis=1)
		nearest_neighbour_table['distance'] = nearest_neighbour_table.id_col.map(self.dist.set_index('rev_id_col')['distance'])
		nearest_neighbour_table['distance_1'] = nearest_neighbour_table['distance']
		nearest_neighbour_table = nearest_neighbour_table.fillna(0)
		nearest_neighbour_table['distance'] = nearest_neighbour_table['distance'] + nearest_neighbour_table['distance_0']
		nearest_neighbour_table = nearest_neighbour_table.loc[nearest_neighbour_table.distance > 0]
		to_merge = nearest_neighbour_table[nearest_neighbour_table.groupby('id_0')\
										  ['distance'].transform(min) == nearest_neighbour_table['distance']]
		to_merge['cluster_fl'] = int(0)
		to_merge = self.check_if_nearest_point_is_part_of_a_cluster(clusterized, to_merge, variables)
		to_merge['cluster_fl_1'] = to_merge['cluster_fl'].astype(int)
		to_merge = to_merge.drop(columns=['cluster_fl', 'point_id', 'id_col', 'distance_0', 'distance', 'distance_1'], axis=1)
		to_merge = self.select_largest_nearest_cluster(to_merge, points)
		points = self.merge_points(points, to_merge)
		return points

	def select_largest_nearest_cluster(self, to_merge, points):
		points_with_more_than_one_nearest_neighbour = set(to_merge['id_0'].loc[to_merge['id_0'].duplicated()==True])
		if len(points_with_more_than_one_nearest_neighbour) > 0:
			for point_w_more_neighbours in points_with_more_than_one_nearest_neighbour:
				to_merge_tmp = to_merge.loc[to_merge.id_0 == point_w_more_neighbours]
				to_merge_tmp['halmaz_meret'] = 0
				to_merge_tmp = to_merge_tmp.reset_index(drop=True)
				for point_to_merge_more_neigh in range(len(to_merge_tmp)):
					to_merge_tmp.iloc[point_to_merge_more_neigh].halmaz_meret = len(points.loc[points.cluster_fl==to_merge_tmp.iloc[point_to_merge_more_neigh].cluster_fl_1])
				to_merge_tmp = to_merge_tmp.sort_values(by=['halmaz_meret'], ascending=False).iloc[0]
				to_merge = to_merge.loc[to_merge.id_0 != point_w_more_neighbours]
				to_merge = pd.concat([to_merge, to_merge_tmp])
				if 'halmaz_meret' in list(to_merge.columns):
					to_merge = to_merge.drop(columns=['halmaz_meret'], axis=1)
		return to_merge

	def commence_clustering(self, points, variables):
		cluster_fl = int(0)
		set_of_flags = {0}
		zeros_now = [0]
		zeros_before = [0,0]
		points = points.reset_index(drop=True)
		self.dist = DistanceMatricesNew().bin_distances(self.dist)
		while (len(zeros_now) < len(zeros_before)) and (0 in set_of_flags):
			self.dist.to_csv('dist_py_1.csv')
			zeros_before = list(points.loc[points.cluster_fl==0].cluster_fl)
			print('zeros before:')
			print(zeros_before)
			[points, cluster_fl] = self.clusterize_nearest(points, cluster_fl)
			self.dist.to_csv('dist_py_2.csv')
			points = self.link_points_which_have_their_nearest_neighbours_in_a_cluster(points, variables)
			self.dist.to_csv('dist_py_3.csv')
			zeros_now = list(points.loc[points.cluster_fl==0].cluster_fl)
			print('zeros now:')
			print(zeros_now)
			set_of_flags = set(points.cluster_fl)
		return points
