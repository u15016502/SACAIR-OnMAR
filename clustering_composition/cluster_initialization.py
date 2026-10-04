import time
from scipy.spatial.distance import cdist
from sklearn.cluster import OPTICS, Birch, BisectingKMeans
from applications.clustering_composition.cluster import cluster
import numpy as np
import random
import math
from sklearn.cluster import MeanShift, DBSCAN, kmeans_plusplus, estimate_bandwidth
from scipy.spatial.distance import cdist
import multiprocessing as mp
import os
from sklearn.cluster import MiniBatchKMeans
from dataset.utils import get_images_for_label, get_image_indices_for_label

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

def initialize_centroids_half_half(data):
	clusters = []

	indices = list(range(0, len(data)))
	random.shuffle(indices)
	slice = math.floor(len(data)/2)

	first_cluster = indices[0:slice]
	second_cluster = indices[slice:]

	c1 = cluster(identifier=None, vectors=[data[idx] for idx in first_cluster], vector_indices=first_cluster)
	c2 = cluster(identifier=None, vectors=[data[idx] for idx in second_cluster], vector_indices=second_cluster)

	return [c1, c2]


def initialize_centroids_kmeans_plus_plus_sillouette_score(data, num_clusters):
	clusters = initialize_centroids_kmeans_plus_plus(data, num_clusters)
	return clusters

def initialize_centroids_kmeans_plus_plus(data, number_of_clusters):
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	flattened = np.array(list(map(lf, data)))
	centers, indices = kmeans_plusplus(flattened, n_clusters=number_of_clusters)
	clusters = [cluster(identifier=None, centroid=center, vectors=[], vector_indices=[]) for idx, center in enumerate(centers)]
	return clusters

def initialize_centroids_birch(data):
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	flattened = np.array(list(map(lf, data)))
	clusters_ = Birch(threshold=random.uniform(0.01,0.25)).fit(flattened)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]


def initialize_centroids_optics(data, distance_metric):
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	flattened = np.array(list(map(lf, data)))
	clusters_ = OPTICS(max_eps=100, metric=get_distance_metric(distance_metric), n_jobs=-1).fit(flattened)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]


def initialize_centroids_bisecting_kmeans(data, num_clusters):
	flattened = np.array([flatten_or_return(dataset_instance, 1) for dataset_instance in data])
	clusters_ = BisectingKMeans(n_clusters=num_clusters, init='k-means++').fit(flattened)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]
