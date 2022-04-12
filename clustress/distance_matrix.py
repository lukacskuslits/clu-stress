import numpy as np

class DistanceMatrices:
    def DistanceMatrix2D(self, df):
        DIST = np.zeros((len(df), len(df)))
        for ii in range(len(df)):
            for jj in range(len(df)):
                d_x_ij = df['x'].iloc[ii] - df['x'].iloc[jj];
                d_y_ij = df['y'].iloc[ii] - df['y'].iloc[jj];
                DIST[ii, jj] = d_x_ij ** 2 + d_y_ij ** 2;

        return DIST

    def DistanceMatrix3D(self, df):
        DIST = np.zeros((len(df), len(df)))
        for ii in range(len(df)):
            for jj in range(len(df)):
                d_x_ij = df['x'].iloc[ii] - df['x'].iloc[jj];
                d_y_ij = df['y'].iloc[ii] - df['y'].iloc[jj];
                d_z_ij = df['z'].iloc[ii] - df['z'].iloc[jj];
                DIST[ii, jj] = d_x_ij ** 2 + d_y_ij ** 2 + d_z_ij ** 2;

        return DIST

    def DistanceMatrix4D(self, df):
        DIST = np.zeros((len(df), len(df)))
        for ii in range(len(df)):
            for jj in range(len(df)):
                d_x_ij = df['x'].iloc[ii] - df['x'].iloc[jj];
                d_y_ij = df['y'].iloc[ii] - df['y'].iloc[jj];
                d_z_ij = df['z'].iloc[ii] - df['z'].iloc[jj];
                d_r_ij = df['r'].iloc[ii] - df['r'].iloc[jj];
                DIST[ii, jj] = d_x_ij ** 2 + d_y_ij ** 2 + d_z_ij ** 2 + d_r_ij ** 2;

        return DIST

    def DiscretizeDistances(self, CluStressObj, DIST):
        # Diszkretizalt bin-ek keszitesehez sorbarendezzuk a tavolsag-ertekeket
        DIST_list = list(DIST.flatten())
        DIST_list.sort()

        # Kozuluk megtartjuk csak az egyedi ertekeket
        set_of_unique_dist_values = set(DIST_list)

        dist_value_list = list(set_of_unique_dist_values)
        dist_value_list.sort()

        # Az adathalmaz diffuzivitasat a tavolsag-ertekek szorasakent definialjuk
        overall_diffusivity = np.std(np.array(dist_value_list))

        collection_of_ids_to_unify = {}
        collection_key = 0
        id_tuple = ()

        # Ciklus: a tavolsag-ertkek egysegbe rendezesehez
        for distance in dist_value_list:
            if distance != 0:
                id_tuple = np.where(((DIST - distance) ** 2 < 100 * overall_diffusivity ** 2) & (DIST != 0))
            if (len(id_tuple) != 0) and (len(id_tuple[0]) != 0):
                collection_of_ids_to_unify.update({collection_key: [id_tuple, distance]})
                collection_key = collection_key + 1
        for collection_key in list(collection_of_ids_to_unify.keys()):
            id_tuple = collection_of_ids_to_unify[collection_key][0]
            distance = collection_of_ids_to_unify[collection_key][1]
            DIST[id_tuple] = distance

        return DIST