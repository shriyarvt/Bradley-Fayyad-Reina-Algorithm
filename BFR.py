import sys
import os
import time
from pyspark import SparkContext
from sklearn.cluster import KMeans
import numpy as np


def load_data(input_f):
    data_raw = sc.textFile(input_f)
    data_split = data_raw.map(lambda line: line.split(','))

    # Expected total length is 12 (10 features + 2 columns for index and cluster)
    expected_total_len = 12
    expected_feature_dim = expected_total_len - 2  # 10 features

    def process_row(x):
        try:
            idx = int(x[0])
            label = int(x[1])

            features = x[2:]

            if len(features) > expected_feature_dim:

                features = features[:expected_feature_dim]
            elif len(features) < expected_feature_dim:
                features.extend(['0'] * (expected_feature_dim - len(features)))

            feature_array = np.array([float(i) for i in features])

            return (idx, label, feature_array)

        except Exception as e:
            print(f"Error processing row: {x}")
            print(f"Error details: {str(e)}")

            return None

    data_rdd = data_split.map(process_row).filter(lambda x: x is not None)

    return data_rdd


def initialize_ds_cs_rs(initial_data, n_cluster):
    initial_data = initial_data.collect()

    # get feature array
    initial_features = np.array([point[2] for point in initial_data])

    # run initial KMeans with 5*n_clusters
    kmeans_1 = KMeans(5 * n_cluster, random_state=55)
    labels_1 = kmeans_1.fit_predict(initial_features)

    # get single-point clusters RS set
    unique, counts = np.unique(labels_1, return_counts=True)
    single_clusters = unique[counts == 1]
    rs_indices = [i for i, label in enumerate(labels_1) if label in single_clusters]
    rs = [initial_data[i] for i in rs_indices]

    # remove RS points for second KMeans
    non_rs_indices = [i for i, label in enumerate(labels_1) if label not in single_clusters]
    non_rs_data = [initial_data[i] for i in non_rs_indices]
    non_rs_features = initial_features[non_rs_indices]

    # second k means
    kmeans_2 = KMeans(n_clusters=n_cluster, random_state=55)
    labels_2 = kmeans_2.fit_predict(non_rs_features)

    # ds stats
    ds = {}
    for i in range(n_cluster):
        cluster_points = non_rs_features[labels_2 == i]
        if len(cluster_points) > 0:
            ds[i] = {
                'N': len(cluster_points),
                'SUM': np.sum(cluster_points, axis=0),
                'SUMSQ': np.sum(np.square(cluster_points), axis=0),
                'points': [p[0] for j, p in enumerate(non_rs_data) if labels_2[j] == i]
            }

    return ds, {}, rs


def compute_mahalanobis(centroid1, cluster_stat2):

    centroid2 = cluster_stat2['SUM'] / cluster_stat2['N']
    variance = (cluster_stat2['SUMSQ'] / cluster_stat2['N']) - np.square(centroid2)
    variance = np.maximum(variance, 1e-6)  # Avoid division by zero
    return np.sqrt(np.sum(np.square((centroid1 - centroid2) / np.sqrt(variance))))


def merge_clusters(cs):

    merged = True
    while merged:
        merged = False
        cs_items = list(cs.items())

        for i, (cs_i, stats_i) in enumerate(cs_items):
            if cs_i not in cs:  # Skip if cluster was already merged
                continue

            # Calculate centroid of first cluster
            centroid_i = stats_i['SUM'] / stats_i['N']
            dimensions = centroid_i.shape[0]
            threshold_dist = 2 * np.sqrt(dimensions)

            # Find closest cluster to merge with
            min_dist = float('inf')
            merge_target = None

            for j, (cs_j, stats_j) in enumerate(cs_items[i + 1:], i + 1):
                if cs_j not in cs:  # Skip if cluster was already merged
                    continue

                # Calculate bidirectional Mahalanobis distance
                dist1 = compute_mahalanobis(centroid_i, stats_j)
                centroid_j = stats_j['SUM'] / stats_j['N']
                dist2 = compute_mahalanobis(centroid_j, stats_i)
                m_dist = min(dist1, dist2)

                # Update merge target if this is the closest valid cluster
                if m_dist < threshold_dist and m_dist < min_dist:
                    min_dist = m_dist
                    merge_target = cs_j

            # Merge clusters if a valid merge target was found
            if merge_target is not None:
                # Update statistics for merged cluster
                stats_j = cs[merge_target]
                merged_n = stats_i['N'] + stats_j['N']
                merged_sum = stats_i['SUM'] + stats_j['SUM']
                merged_sumsq = stats_i['SUMSQ'] + stats_j['SUMSQ']

                # Update cluster i with merged information
                cs[cs_i]['N'] = merged_n
                cs[cs_i]['SUM'] = merged_sum
                cs[cs_i]['SUMSQ'] = merged_sumsq
                cs[cs_i]['points'].extend(cs[merge_target]['points'])

                # Remove merged cluster
                del cs[merge_target]
                merged = True
                break

    return cs


def output_stats(ds, cs, rs):
    n_discard_points = sum(cluster['N'] for cluster in ds.values())
    n_clusters_cs = len(cs) if isinstance(cs, dict) else 0
    n_compression_points = sum(cluster['N'] for cluster in cs.values()) if isinstance(cs, dict) else 0
    n_retained_points = len(rs)
    return n_discard_points, n_clusters_cs, n_compression_points, n_retained_points


def write_intermediate_results(output_file, rounds_data, ds, cs, rs):
    with open(output_file, 'w') as f:

        f.write("The intermediate results:\n")
        for key, stats in sorted(rounds_data.items()):
            f.write(f"{key}: {stats[0]},{stats[1]},{stats[2]},{stats[3]}\n")

        f.write("\nThe clustering results:\n")

        final_clusters = {}

        for cluster_id, stats in ds.items():
            for point_idx in stats['points']:
                final_clusters[point_idx] = cluster_id

        for cs_stats in cs.values():
            for point_idx in cs_stats['points']:
                final_clusters[point_idx] = -1

        for point in rs:
            final_clusters[point[0]] = -1

        for point_idx in sorted(final_clusters.keys()):
            f.write(f"{point_idx},{final_clusters[point_idx]}\n")


def process_chunk(chunk_points, ds, cs, rs, n_cluster):
    chunk_data = chunk_points.collect()
    new_rs = []

    # steps 8-10
    for point in chunk_data:
        assigned = False

        min_ds_dist = float('inf')
        best_ds = None
        for ds_id, ds_stats in ds.items():
            dist = compute_mahalanobis(point[2], ds_stats)
            if dist < min_ds_dist and dist < 2 * np.sqrt(point[2].shape[0]):
                min_ds_dist = dist
                best_ds = ds_id

        if best_ds is not None:
            # update DS
            ds[best_ds]['N'] += 1
            ds[best_ds]['SUM'] += point[2]
            ds[best_ds]['SUMSQ'] += np.square(point[2])
            ds[best_ds]['points'].append(point[0])
            assigned = True
            continue

        if not assigned and cs:
            min_cs_dist = float('inf')
            best_cs = None
            for cs_id, cs_stats in cs.items():
                dist = compute_mahalanobis(point[2], cs_stats)
                if dist < min_cs_dist and dist < 2 * np.sqrt(point[2].shape[0]):
                    min_cs_dist = dist
                    best_cs = cs_id

            if best_cs is not None:
                cs[best_cs]['N'] += 1
                cs[best_cs]['SUM'] += point[2]
                cs[best_cs]['SUMSQ'] += np.square(point[2])
                cs[best_cs]['points'].append(point[0])
                assigned = True
                continue

        if not assigned:
            new_rs.append(point)

    rs.extend(new_rs)

    # step 11
    if len(rs) > 0:
        rs_features = np.array([point[2] for point in rs])
        kmeans_rs = KMeans(n_clusters=min(5 * n_cluster, len(rs)), random_state=55)
        labels_rs = kmeans_rs.fit_predict(rs_features)

        unique, counts = np.unique(labels_rs, return_counts=True)

        single_clusters = unique[counts == 1]
        new_rs = [rs[i] for i, label in enumerate(labels_rs) if label in single_clusters]

        multi_clusters = unique[counts > 1]
        cs_id = len(cs)
        for cluster_id in multi_clusters:
            cluster_points = rs_features[labels_rs == cluster_id]
            cs[cs_id] = {
                'N': len(cluster_points),
                'SUM': np.sum(cluster_points, axis=0),
                'SUMSQ': np.sum(np.square(cluster_points), axis=0),
                'points': [rs[i][0] for i, label in enumerate(labels_rs) if label == cluster_id]
            }
            cs_id += 1
        rs = new_rs

    # step 12- CHECK
    if cs:
        cs = merge_clusters(cs)

    return ds, cs, rs


def merge_cs_with_ds(cs, ds):

    if not cs:
        return ds, cs

    cs_items = list(cs.items())
    for cs_id, cs_stats in cs_items:
        if cs_id not in cs:
            continue

        min_dist = float('inf')
        best_ds = None
        cs_centroid = cs_stats['SUM'] / cs_stats['N']
        dimensions = cs_centroid.shape[0]
        threshold_dist = 2 * np.sqrt(dimensions)

        for ds_id, ds_stats in ds.items():

            dist1 = compute_mahalanobis(cs_centroid, ds_stats)
            ds_centroid = ds_stats['SUM'] / ds_stats['N']
            dist2 = compute_mahalanobis(ds_centroid, cs_stats)
            m_dist = min(dist1, dist2)


            if m_dist < min_dist and m_dist < threshold_dist:
                min_dist = m_dist
                best_ds = ds_id

        if best_ds is not None:

            ds[best_ds]['N'] += cs_stats['N']
            ds[best_ds]['SUM'] += cs_stats['SUM']
            ds[best_ds]['SUMSQ'] += cs_stats['SUMSQ']
            ds[best_ds]['points'].extend(cs_stats['points'])

            del cs[cs_id]

    return ds, cs

if __name__ == '__main__':
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

    start = time.time()

    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    # initialize spark
    sc = SparkContext('local[*]', 'BFR')
    sc.setLogLevel("ERROR")

    # load and split data
    data = load_data(input_file)
    initial_data, remaining_data = data.randomSplit([0.2, 0.8], seed=55)

    # initial sets
    ds, cs, rs = initialize_ds_cs_rs(initial_data, n_cluster)

    if len(rs) > 0:
        rs_features = np.array([point[2] for point in rs])
        kmeans_3 = KMeans(n_clusters=min(5 * n_cluster, len(rs)), random_state=55)
        labels_3 = kmeans_3.fit_predict(rs_features)

        unique, counts = np.unique(labels_3, return_counts=True)

        # single point clusters stay in RS
        single_clusters = unique[counts == 1]
        new_rs = [rs[i] for i, label in enumerate(labels_3) if label in single_clusters]

        # otherwise in CS
        multi_clusters = unique[counts > 1]
        cs = {}
        for i, cluster_id in enumerate(multi_clusters):
            cluster_points = rs_features[labels_3 == cluster_id]
            cs[i] = {
                'N': len(cluster_points),
                'SUM': np.sum(cluster_points, axis=0),
                'SUMSQ': np.sum(np.square(cluster_points), axis=0),
                'points': [rs[j][0] for j, label in enumerate(labels_3) if label == cluster_id]
            }
        rs = new_rs

    # round 1 stats
    round1_stats = output_stats(ds, cs, rs)
    rounds_intermediate_data = {'Round 1': list(round1_stats)}

    # remaining data
    chunks = remaining_data.randomSplit([0.25, 0.25, 0.25, 0.25], seed=55)

    for round_num, chunk in enumerate(chunks, 2):

        ds, cs, rs = process_chunk(chunk, ds, cs, rs, n_cluster)

        # get statistics
        round_stats = output_stats(ds, cs, rs)
        rounds_intermediate_data[f'Round {round_num}'] = list(round_stats)

    # merge CS with DS
    ds, cs = merge_cs_with_ds(cs, ds)

    # write results
    write_intermediate_results(output_file, rounds_intermediate_data, ds, cs, rs)

    end = time.time()
    print('Duration:', str(end - start))

    sc.stop()