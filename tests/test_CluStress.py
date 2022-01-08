import pandas as pd
import numpy as np
from unittest import TestCase
from codes.clu_stress import CluStress


class TestCluStress(TestCase):

    mock_pontok = pd.DataFrame({'x': [1,2], 'y': [3,4]})
    clu_stress = CluStress('2D', mock_pontok)

    mock_mindices0 = [0, 0, 0, 1, 1, 7, 7, 7, 5, 5, 5]
    mock_mindices1 = [2, 5, 1, 4, 3, 9, 8, 7, 6, 5, 4]

    def test_group_connected_pairs_of_min_dist(self):
        self.clu_stress.group_connected_pairs_of_minimum_distance()

    def test_group_all_connected_pairs_of_min_dist(self):
        check_dict = {0: [np.array([[0, 2],\
                     [0, 5],\
                     [0, 1],\
                     [5, 6],\
                     [5, 5],\
                     [5, 4],\
                     [1, 4],\
                     [1, 3]])], 1: [np.array([[0, 2],\
                     [0, 5],\
                     [0, 1],\
                     [5, 6],\
                     [5, 5],\
                     [5, 4],\
                     [1, 4],\
                     [1, 3]])], 2: [np.array([[7, 9],\
                     [7, 9],\
                     [7, 8],\
                     [7, 7]])]}
        returned_dict = self.clu_stress.group_all_connected_pairs_of_min_dist(
                        self.mock_mindices0, self.mock_mindices1)
        print(returned_dict)
        self.assertListEqual(list(returned_dict), list(check_dict))

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

