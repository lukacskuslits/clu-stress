import pandas as pd
from unittest import TestCase
from src.clustress.clu_stress import CluStress


class TestCluStressNearestPoints(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_group_connected_pairs_of_min_dist(self):
        check_cluster_fl = [1, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 0]
        pontok, cluster_fl = self.clu_stress.clusterize_nearest(self.mock_points, self.mock_cluster_fl)
        self.assertListEqual(check_cluster_fl, list(pontok.cluster_fl))


class TestCluStressAllNearestPoints(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_group_all_connected_pairs_of_min_dist(self):
        check_cluster_fl = dict({0: {'cluster_fl': 1}, 1: {'cluster_fl': 0}, 2: {'cluster_fl': 1}, 3: {'cluster_fl': 2},
                                 4: {'cluster_fl': 2}, 5: {'cluster_fl': 3}, 6: {'cluster_fl': 3}, 7: {'cluster_fl': 3},
                                 8: {'cluster_fl': 4}, 9: {'cluster_fl': 4}, 10: {'cluster_fl': 4}, 11: {'cluster_fl': 0}})
        pontok, cluster_fl = self.clu_stress.clusterize_nearest(self.mock_points.sample(frac=1), self.mock_cluster_fl)
        pontok.index = pontok.PID
        pontok = pontok.sort_index()
        self.assertEqual(check_cluster_fl, pontok[['cluster_fl']].to_dict(orient='index'))


class TestCluStressConnPoints(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_get_connected_points(self):
        mock_nearest_point_pairs = [(0,2), (3,4), (5,6), (6,7), (8,9), (9,10)]
        connected_point_pairs = self.clu_stress.get_connected_points(mock_nearest_point_pairs)
        print(connected_point_pairs)


class TestCluStressCheckPoints(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_check_if_nearest_point_is_part_of_a_cluster(self):
        mock_points_tmp = self.mock_points.copy()
        mock_points_tmp['cluster_fl'] = [1, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 0]
        mock_connected_points = pd.DataFrame({'id_col': [(1,3)], 'id_0': [1], 'id_1': [3], 'distance_0': [2.0], 'distance': [2.0], 'distance_1': [0.0]})
        mock_connected_points['cluster_fl'] = 0
        mock_clusterized = mock_points_tmp.loc[mock_points_tmp['cluster_fl'] != 0]
        to_merge = self.clu_stress.check_if_nearest_point_is_part_of_a_cluster(mock_clusterized, mock_connected_points, self.mock_variables)
        print(to_merge)


class TestCluStressLinkPoints(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_link_points_which_have_their_nearest_neighbours_in_a_cluster(self):
        self.clu_stress.link_points_which_have_their_nearest_neighbours_in_a_cluster(self.mock_points, self.mock_variables)


class TestCluStressMergePoints(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_merge_points(self):
        mock_to_merge = pd.DataFrame({'id_col': [(1,3)], 'id_0': [1], 'id_1': [3], 'distance_0': [2.0], 'distance': [2.0], 'distance_1': [0.0]})
        mock_to_merge['cluster_fl'] = 2
        mock_to_merge['cluster_fl_1'] = mock_to_merge['cluster_fl']
        mock_to_merge = mock_to_merge.drop(columns=['cluster_fl', 'id_col', 'distance_0', 'distance', 'distance_1'],
                                 axis=1)
        mock_to_merge = self.clu_stress.select_largest_nearest_cluster(mock_to_merge, self.mock_points)
        points = self.clu_stress.merge_points(self.mock_points, mock_to_merge)
        print(points)


class TestCluStressFullClustering(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_commence_clustering(self):
        points = self.clu_stress.commence_clustering(self.mock_points, self.mock_variables)
        print(points)
        # import matplotlib.pyplot as plt
        # plt.scatter(points['x'], points['y'], c=points.cluster_fl)
        # plt.show()


class TestCluStressFullClustering_b(TestCase):
    mock_mindices0 = [0, 1, 2, 3, 5, 8, 8, 8, 8, 14]
    mock_mindices1 = [3, 1, 3, 0, 3, 3, 2, 4, 6, 8]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_commence_clustering_b(self):
        [points, points_outlier, _] = self.clu_stress.select_outliers(self.mock_points)
        points = self.clu_stress.commence_clustering(self.mock_points, self.mock_variables)
        print(points)
        import matplotlib.pyplot as plt
        plt.scatter(points['x'], points['y'], c=points.cluster_fl)
        plt.show()

class TestBlobs(TestCase):
    from sklearn.datasets import make_blobs, make_moons
    var, clust = make_blobs(n_samples=100, centers=3, n_features=2)
    #var, clust = make_moons(n_samples=100, noise=0.1)
    mock_variables = ['x', 'y']

    def test_blobs(self):
        import numpy as np
        points = pd.DataFrame({'x': list(self.var[:, 0]), 'y': list(self.var[:, 1]), 'cluster_fl': np.zeros((1,100)).tolist()[0], 'cluster_ref': list(self.clust)})
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        for column in self.mock_variables:
            points[column] = scaler.fit_transform(points[column].values.reshape(-1, 1))
        points['PID'] = points.index
        print(points.head())
        clu_stress = CluStress(points[['x','y', 'cluster_fl', 'PID']])
        [points, points_outlier, _] = clu_stress.select_outliers(points)
        points = clu_stress.commence_clustering(points[['x', 'y', 'cluster_fl', 'PID']], self.mock_variables)
        print(points)
        import matplotlib.pyplot as plt
        points = pd.concat([points_outlier, points], axis=0)
        n_clusters_h = len(points.cluster_fl.unique())
        fig, ax = plt.subplots()
        scatter = ax.scatter(points['x'], points['y'], c=points.cluster_fl)
        legend = ax.legend(*scatter.legend_elements(num=n_clusters_h), loc='lower right', title='Blob clusters',
                           prop={'size': 4})
        ax.add_artist(legend)
        plt.show()


class TestOutliers(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 9, 30, 15]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 10, 50, 25]
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11, 12, 13]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for column in mock_points.columns:
        mock_points[column] = scaler.fit_transform(mock_points[column].values.reshape(-1, 1))
    mock_points['PID'] = list(mock_points.index)
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mock_variables = ['x', 'y']
    clu_stress = CluStress(mock_points)

    def test_outliers(self):
        [points, points_outlier, _] = self.clu_stress.select_outliers(self.mock_points)
        points = pd.concat([points_outlier, points], axis=0)
        import matplotlib.pyplot as plt
        plt.scatter(points['x'], points['y'], c=points.cluster_fl)
        plt.show()