from scipy.spatial.distance import cdist
import numpy as np
from scipy.cluster.hierarchy import ward
import concurrent.futures
from copy import deepcopy
import random
import math
from scipy.spatial.distance import pdist
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import multiprocessing as mp
import os
from sklearn.cluster import MiniBatchKMeans

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

def remove_cluster_largest_inter_distance(data, clusters, distance_metric):
	if len(clusters) == 0 or len(data) == len(clusters) or len(clusters) == 1:
		return clusters

	largest_distance_cluster_distance = 0
	largest_distance_cluster = []

	for c in clusters:
		if len(c.vectors) > 0:
			def get_distance(cluster_):
				d = 0
				for idx, v in enumerate(cluster_.vectors):
					if idx + 1 == len(cluster_.vectors):
						return np.average(d)
					else:
						
						d += apply_distance_metric(distance_metric, flatten_or_return(v, 2), flatten_or_return(cluster_.vectors[idx + 1], 2))
				

			cluster_distance = get_distance(c)


			if cluster_distance > largest_distance_cluster_distance:
				largest_distance_cluster_distance = cluster_distance
				largest_distance_cluster = c.identifier


	return [c_ for c_ in clusters if c_.identifier not in [largest_distance_cluster]]


def remove_cluster_most_dataset_instances(data, clusters, distance_metric):
	if len(clusters) == 0 or len(clusters) == len(data):
		return clusters

	largest_number_instances = 0
	most_instances_cluster = []

	for c in clusters:
		num_instances = len(c.vectors)

		if num_instances > largest_number_instances:
			largest_number_instances = num_instances
			most_instances_cluster = [c.identifier]

	return [c_ for c_ in clusters if c_.identifier not in [most_instances_cluster]]
	
def remove_cluster_least_dataset_instances(data, clusters, distance_metric):
	if len(clusters) == 0 or len(clusters) == len(data):
		return clusters
		
	smallest_number_instances = None
	least_instances_cluster = []

	for c in clusters:
		num_instances = len(c.vectors)

		if smallest_number_instances == None or num_instances < smallest_number_instances:
			smallest_number_instances = num_instances
			least_instances_cluster = [c.identifier]

	return [c_ for c_ in clusters if c_.identifier not in [least_instances_cluster]]

def remove_cluster_random(data, clusters, distance_metric):
	if len(clusters) == 0:
		return []

	random_list = list(range(0, len(clusters)))
	random.shuffle(random_list)
	selection = random_list[0]
	clusters.pop(selection)

	return clusters

def check_for_overlap(v1, v2):
	combined = np.concatenate((v1, v2), axis=0)
	if len(combined) > len(np.unique(combined)):
		return True
	else:
		return False

def remove_overlapping_clusters(data, clusters):
	if len(data) == len(clusters) or len(clusters) == 1:
		return clusters

	smallest_overlapping_clusters = []

	for c1 in clusters:
		for c2 in clusters:
			if c1.identifier != c2.identifier and len(c1.vectors) > 0 and len(c2.vectors) > 0:
				if check_for_overlap(c1.vectors, c2.vectors) == True:
					if len(c1.vectors) < len(c2.vectors):
						smallest_overlapping_clusters.append(c1.identifier)
					else:
						smallest_overlapping_clusters.append(c2.identifier)


	return [c_ for c_ in clusters if c_.identifier not in smallest_overlapping_clusters]
