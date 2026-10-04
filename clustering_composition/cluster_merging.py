from applications.clustering_composition.cluster import cluster
import numpy as np
import random
from scipy.spatial.distance import cdist 

# operates on a single image
def flatten_or_return(vec, expected_shape=1):

	if len(vec.shape) == 1 and expected_shape == 1:
		return vec
	elif len(vec.shape) == 2 and expected_shape == 1:
		return vec.flatten()
	elif len(vec.shape) == 2 and expected_shape == 2:
		return vec
	elif len(vec.shape) == 3 and expected_shape == 1:
		return vec.flatten()
	elif len(vec.shape) == 3 and expected_shape == 2:
		return np.array(vec.flatten()).reshape(1, -1)
	elif len(vec.shape) == 1 and expected_shape == 2:
		return np.array(vec).reshape(1, -1)
	else:
		return vec.flatten()

def concat_or_return(a, b):
	if len(a) == 0:
		return b

	if len(b) == 0:
		return a

	return np.unique(
			np.concatenate(
				(a, b),
				axis=0
			)
		, axis=0)

def get_distance_metric(distance_metric):

	if distance_metric == 0: # Euclidean
		return 'euclidean'
	if distance_metric == 1: # Manhattan
		return 'manhattan'
	if distance_metric == 2: # Minkowski
		return 'minkowski'
	if distance_metric == 3: # Hamming
		return 'hamming'
	if distance_metric == 4: # Cosine
		return 'cosine'
	if distance_metric == 5: # Chebyshev
		return 'chebyshev'
	if distance_metric == 6: # Canberra
		return 'canberra'
	if distance_metric == 7:
		return 'euclidean'

# operates on a single image
def apply_distance_metric(distance_metric, a, b):
	# a = np.array_split(a, 1)
	# b = np.array_split(b, 1)

	dis = 0

	if np.array(a).shape != np.array(b).shape:
		if np.array(a).shape[0] == 1:
			b = np.array(b).reshape(a.shape)
		else:
			a = np.array(a).reshape(b.shape)

	if distance_metric == 0: # Euclidean
		dis = np.mean(np.array(cdist(a, b, 'euclidean')))
	elif distance_metric == 1: # Manhattan
		dis = np.mean(np.array(cdist(a, b, 'cityblock')))
	elif distance_metric == 2: # Minkowski
		dis = np.mean(np.array(cdist(a, b, 'minkowski')))
	elif distance_metric == 3: # Hamming
		dis =  np.mean(np.array(cdist(a, b, 'hamming')))
	elif distance_metric == 4: # Cosine
		dis =  np.mean(np.array(cdist(a, b, 'cosine')))
	elif distance_metric == 5: # Chebyshev
		dis =  np.mean(np.array(cdist(a, b, 'chebyshev')))
	elif distance_metric == 6: # L2
		dis =  np.mean(np.array(cdist(a, b, 'canberra')))
	else:
		dis = np.mean(np.array(cdist(a, b, 'euclidean')))
		dis += np.mean(np.array(cdist(a, b, 'cityblock')))
		dis += np.mean(np.array(cdist(a, b, 'minkowski')))
		dis += np.mean(np.array(cdist(a, b, 'hamming')))
		dis += np.mean(np.array(cdist(a, b, 'cosine')))
		dis += np.mean(np.array(cdist(a, b, 'chebyshev')))
		dis += np.mean(np.array(cdist(a, b, 'canberra')))
		dis /=7

	# for idx, x in enumerate(a):
	# 	temp_distance = 0

	# 	if distance_metric == 0: # Euclidean
	# 		temp_distance = distance.euclidean(a[sc], b[idx])
	# 	elif distance_metric == 1: # Manhattan
	# 		temp_distance = distance.cityblock(a[idx], b[idx])
	# 	elif distance_metric == 2: # Minkowski
	# 		temp_distance = distance.minkowski(a[idx], b[idx])
	# 	elif distance_metric == 3: # Hamming
	# 		temp_distance = distance.hamming(a[idx], b[idx])
	# 	elif distance_metric == 4: # Cosine
	# 		temp_distance = distance.cosine(a[idx], b[idx])
	# 	elif distance_metric == 5: # Chebyshev
	# 		temp_distance = distance.chebyshev(a[idx], b[idx])
	# 	elif distance_metric == 6: # L2
	# 		temp_distance = distance.canberra(a[idx], b[idx])
	# 	else:
	# 		temp_distance = distance.euclidean(a[idx], b[idx])
	# 		temp_distance += distance.cityblock(a[idx], b[idx])
	# 		temp_distance += distance.minkowski(a[idx], b[idx])
	# 		temp_distance += distance.hamming(a[idx], b[idx])
	# 		temp_distance += distance.cosine(a[idx], b[idx])
	# 		temp_distance += distance.chebyshev(a[idx], b[idx])
	# 		temp_distance += distance.canberra(a[idx], b[idx])

	# 		temp_distance /= 7

	return dis

def merge_clusters_random(data, clusters, distance_metric):
	if len(clusters) <= 2:
		return clusters

	indices = list([idx for idx, c in enumerate(clusters) if len(c.vectors) > 0])
	
	if len(indices) == 0 or len(indices) == 1:
		return clusters

	random.shuffle(indices)
	cluster_1 = clusters[indices[0]]
	cluster_2 = clusters[indices[1]]

	merging_clusters = [cluster_1.identifier, cluster_2.identifier]
	remaining_clusters = [c_ for c_ in clusters if c_.identifier not in merging_clusters]
	merged_cluster = cluster(cluster_1.identifier, [], concat_or_return(cluster_1.vectors, cluster_2.vectors), concat_or_return(cluster_1.vector_indices, cluster_2.vector_indices))
	return remaining_clusters + [merged_cluster]

def merge_clusters_with_least_dataset_instances(data, clusters, distance_metric):
	if len(clusters) <= 1:
		return clusters

	if len(clusters) == 2:
		return [cluster(identifier=None, vectors=data, vector_indices=range(0,len(data)))]

	ordered_clusters = []

	for idx, c in enumerate(clusters):
		if len(c.vectors) > 0:
			ordered_clusters.append({
				'num_instances': len(c.vectors),
				'cluster': c
			})
	
	if len(ordered_clusters) == 0 or len(ordered_clusters) == 1:
		return clusters	

	ordered_clusters.sort(key=lambda x: x['num_instances'], reverse=False)

	cluster_1 = ordered_clusters[0]['cluster']
	cluster_2 = ordered_clusters[1]['cluster']

	merging_clusters = [cluster_1.identifier, cluster_2.identifier]
	remaining_clusters = [c_ for c_ in clusters if c_.identifier not in merging_clusters]
	merged_cluster = cluster(cluster_1.identifier, [], concat_or_return(cluster_1.vectors, cluster_2.vectors), concat_or_return(cluster_1.vector_indices, cluster_2.vector_indices))
	return remaining_clusters + [merged_cluster]

def merge_clusters_by_distance(data, clusters, distance_metric):
	if len(clusters) <= 2:
		return clusters

	smallest_distance = []
	smallest_cluster_1 = None
	smallest_cluster_2 = None
	points = []
	clusters_ = []

	for c1 in clusters:
		if len(c1.vectors) > 0:
			clusters_.append(c1)
			points.append(flatten_or_return(c1.centroid, 1))

	dists = cdist(points, points, metric=get_distance_metric(distance_metric).replace('manhattan', 'cityblock'))
	dists[dists == 0] = dists.max()
	smallests = np.unravel_index(dists.argmin(), dists.shape)
	
	smallest_cluster_1 = clusters_[smallests[0]]
	smallest_cluster_2 = clusters_[smallests[1]]

	smallest_clusters = [smallest_cluster_1.identifier, smallest_cluster_2.identifier]

	remaining_clusters = [c_ for c_ in clusters if c_.identifier not in smallest_clusters]
	
	merged_cluster = cluster(smallest_cluster_1.identifier, [], concat_or_return(smallest_cluster_1.vectors, smallest_cluster_2.vectors), concat_or_return(smallest_cluster_1.vector_indices, smallest_cluster_2.vector_indices))
	
	return remaining_clusters + [merged_cluster]

