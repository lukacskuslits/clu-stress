import pandas as pd
import numpy as np
from unittest import TestCase
from src.clustress.clu_stress import CluStress


class TestCluStress(TestCase):
    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5, 20]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4, 13]
    mock_points = pd.DataFrame({'x': mock_mindices0, 'y': mock_mindices1})
    mock_points['PID'] = list(mock_points.index)
    mock_cluster_fl = 0
    mock_points['cluster_fl'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    clu_stress = CluStress(mock_points)




    def test_group_connected_pairs_of_min_dist(self):
        check_cluster_fl = [1, 0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 0]
        pontok, cluster_fl = self.clu_stress.clusterize_nearest(self.mock_points, self.mock_cluster_fl)
        self.assertListEqual(check_cluster_fl, list(pontok.cluster_fl))

    def test_group_all_connected_pairs_of_min_dist(self):
        check_cluster_fl = dict({0: {'cluster_fl': 1}, 1: {'cluster_fl': 0}, 2: {'cluster_fl': 1}, 3: {'cluster_fl': 2},
                                 4: {'cluster_fl': 2}, 5: {'cluster_fl': 3}, 6: {'cluster_fl': 3}, 7: {'cluster_fl': 3},
                                 8: {'cluster_fl': 4}, 9: {'cluster_fl': 4}, 10: {'cluster_fl': 4}, 11: {'cluster_fl': 0}})
        pontok, cluster_fl = self.clu_stress.clusterize_nearest(self.mock_points.sample(frac=1), self.mock_cluster_fl)
        pontok.index = pontok.PID
        pontok = pontok.sort_index()
        self.assertEqual(check_cluster_fl, pontok[['cluster_fl']].to_dict(orient='index'))

    def test_clusterize_closest(self):
        self.clu_stress.ClusterizeClosest()

    def test_get_connected_points(self):
        self.clu_stress.get_connected_points(mock_nearest_point_pairs)

    def test_check_if_closest_point_clusterized_point(self):
        self.clu_stress.check_if_nearest_point_is_part_of_a_cluster(mock_clusterized, mock_connected_points, mock_variables)

    def test_merge_points(self):
        self.clu_stress.merge_points(pontok, to_merge)

    def test_commence_clustering(self):
        self.clu_stress.commence_clustering()

