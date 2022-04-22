import itertools
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class DistanceMatricesNew:

    #TODO: this method works only with scaled coordinate values,
    # needs to be refactored to handle dimensional values as well
    # outlier ranges need to be computed for the full distance matrix
    def compute_distance_matrix(self, df, variables):
        DIST = np.zeros((len(df), len(df)))
        var = 0
        for variable in variables:
            coord_values_arr = df[variable].values
            coord_values_arr = np.tile(coord_values_arr, (len(coord_values_arr), 1))
            coord_values_arr_tr = coord_values_arr.T
            DIST += (coord_values_arr - coord_values_arr_tr)**2
            var = var + 1
        return DIST

    # converts distance_matrix to a sparse array dictionary of keys format
    def transform_distance_matrix(self, distance_matrix):
        dist_sparse = csr_matrix(np.triu(distance_matrix).copy())
        dist_dict = dist_sparse.todok()
        dist_dict = dict(dist_dict.items())
        dist_df = pd.DataFrame({'id_col': list(dist_dict.keys()), 'distance': list(dist_dict.values())})
        dist_df['id_0'] = dist_df['id_col'].apply(lambda x: x[0])
        dist_df['id_1'] = dist_df['id_col'].apply(lambda x: x[1])
        dist_df['id_0_r'] = dist_df['id_col'].apply(lambda x: x[1])
        dist_df['id_1_r'] = dist_df['id_col'].apply(lambda x: x[0])
        return dist_df

    # keeps only those elements of the distance matrix which are not clusterized
    def set_distance_matrix(self, dist, clusterized_list):
        all_clusterized_pairs = list(itertools.combinations(clusterized_list, 2))
        dist = dist.loc[np.invert(dist.id_col.isin(all_clusterized_pairs))]
        return dist

    def bin_distances(self, dist):
        dist = dist.sort_values(by='distance')
        var_bins = np.histogram_bin_edges(dist['distance'].values, bins='auto')
        #Interpolate between bins (resampling)
        for step in range(1):
            n = len(var_bins)
            var_bin_range = np.arange(0, 2*n, 2)
            var_bin_range_new = np.arange(2*n-1)
            var_bins = np.interp(var_bin_range_new, var_bin_range, var_bins)
        print(len(var_bins))
        dist['distance'] = list(np.digitize(dist['distance'].values, var_bins))
        dist = dist.sort_values(by='id_0')
        return dist

    def find_nearest_dists(self, dist):
        """Finds nearest distances for in the distance matrix to determine outlier ranges."""
        nearest_dists = []
        for row in range(dist.shape[0]):
            dist_i = dist[row,:]
            nearest_dists.append(np.min(dist_i[np.nonzero(dist_i)]))
        nearest_dists = pd.DataFrame({'distance': nearest_dists})
        return nearest_dists

    def flip_values(self, tuple_of_vals):
        flipped_list = [0,0]
        flipped_list[1] = tuple_of_vals[0]
        flipped_list[0] = tuple_of_vals[1]
        flipped_tuple = tuple(flipped_list)
        return flipped_tuple


