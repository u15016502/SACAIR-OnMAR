from sklearn.decomposition import PCA
import markov_clustering as mc
import networkx as nx
import kmedoids
from scipy.cluster.vq import vq
import time
from scipy.spatial.distance import cdist
import gc
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.cluster import AgglomerativeClustering, FeatureAgglomeration, MeanShift, DBSCAN, AffinityPropagation, kmeans_plusplus, SpectralClustering, OPTICS, Birch, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from applications.clustering_composition.cluster import cluster
import numpy as np
from scipy.spatial.distance import cdist
import multiprocessing as mp
import os
from sklearn.cluster import MiniBatchKMeans
from dataset.utils import get_images_for_label, get_image_indices_for_label
from copy import deepcopy

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

def get_images_for_label(label, images, labels):
	_images = []

	for idx, image in enumerate(images):
		if labels[idx] == label:
			_images.append(deepcopy(image))
		else:
			if str(labels[idx]) == str(label):
				_images.append(deepcopy(image))

	return _images

def get_image_indices_for_label(label, images, labels):
	_images = []

	for idx, image in enumerate(images):
		if labels[idx] == label:
			_images.append(deepcopy(idx))
		else:
			if str(labels[idx]) == str(label):
				_images.append(deepcopy(idx))

	return _images

def mean_shift(data, clusters, membership_limit, num_clusters):
	
	min_samples = np.array(data).shape[1] * 2
 
	if min_samples > len(data):
		min_samples = len(data)

	lf = lambda dataset_instance: flatten_or_return(dataset_instance)
	data_ = np.array(list(map(lf, data)))

	mn = MeanShift(n_jobs=-1)
	clusters_ = mn.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]

def dbscan(data, clusters, membership_limit, distance_metric, num_clusters):

	min_samples = data.shape[1] * 2

	if min_samples > len(data):
		min_samples = len(data)
	
	lf = lambda dataset_instance: flatten_or_return(dataset_instance)
	data_ = np.array(list(map(lf, data)))

	neighbors = NearestNeighbors(n_neighbors=min_samples)
	neighbors_fit = neighbors.fit(data_)
	gc.collect()
	distances, indices = neighbors_fit.kneighbors(data_)

	distances = np.sort(distances, axis=0)
	distances = distances[:,1]

	num_clusters = range(1, len(distances)+1)

	kn = KneeLocator(num_clusters, distances, curve='convex', direction='increasing')

	if kn.knee == len(distances):
		eps = distances[len(distances) - 1]
	else:
		eps = distances[kn.knee]

	p_val = None
	
	if distance_metric == 2:
		p_val = 1

	dbs = DBSCAN(eps = eps, min_samples=min_samples, metric=get_distance_metric(distance_metric), p=p_val)
	clusters_ = dbs.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)

	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]


def exemplar(data, clusters, membership_limit):

	ap = AffinityPropagation()
	gc.collect()

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))

	clusters_ = ap.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]

def minibatch(data, clusters, distance_metric, membership_limit, num_clusters):
	kmeans = MiniBatchKMeans(n_clusters = num_clusters, max_no_improvement=5)
	gc.collect()

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))

	clusters_ = kmeans.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]


def kmediods(data, clusters, distance_metric, membership_limit, num_clusters):
	medoids = num_clusters

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))
	# diss = pairwise_distances(data_, metric=get_distance_metric(distance_metric))

	clusters_ = kmedoids.fasterpam(data_, medoids=int(medoids), max_iter=2, init='random')
	clusters_.labels = clusters_.labels.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels), get_image_indices_for_label(label, data, clusters_.labels)) for label in new_unique_labels]

def markov_clustering(data, clusters, distance_metric, membership_limit):
	numnodes = len(data)

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))

	positions = {idx: (
					np.min(i),
					np.max(i)
				) for idx, i in enumerate(data_)}

	network = nx.random_geometric_graph(numnodes, 0.3, pos=positions)

	matrix = nx.to_scipy_sparse_matrix(network)

	for inflation in [i / 10 for i in range(15, 26)]:
		result = mc.run_mcl(matrix, inflation=inflation)
		clusters_ = mc.get_clusters(result)
		Q = mc.modularity(matrix=result, clusters=clusters_)

	result = mc.run_mcl(matrix, inflation=2.1)
	clusters_ = mc.get_clusters(result)
	unique_labels = range(0,len(clusters_))
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	new_clustering = []

	for idx1, dataset_instance in enumerate(data):
		for idx2, arr in enumerate(clusters_):
			if idx1 in arr: 
				new_clustering.append(new_unique_labels[idx2])

	unique_labels = np.unique(new_clustering)

	new_clusters = [cluster(label, [], get_images_for_label(label, data, new_clustering), get_image_indices_for_label(label, data, new_clustering)) for label in new_unique_labels]

	if membership_limit == 0:
		return new_clusters
	else:
		return new_clusters + clusters

def convert_clusters_to_label_representations(data, clusters, membership_limit, distance_metric):
	labels = []

	if len(clusters) == 0:
		return [-1] * len(data)

	if membership_limit == 0:
		for idx, dataset_instance in enumerate(data):
			label = -1
			if clusters != None:
				for c_ in clusters:
					if c_.check_for_vector(idx) == True:
						label = c_.identifier
						break

			labels.append(label)
	else:
		for idx, dataset_instance in enumerate(data):
			label = -1
			temp_lab = []
			temp_dist = []
			if clusters != None:
				for c_ in clusters:
					if c_.check_for_vector(idx) == True:
						temp_lab.append(c_.identifier)
						temp_dist.append(flatten_or_return(c_.centroid, 1))
				
				if len(temp_lab) > 0:
					dists = cdist([flatten_or_return(dataset_instance, 1)], temp_dist, metric=get_distance_metric(distance_metric).replace('manhattan', 'cityblock'))
					dists[dists == 0] = dists.max()
					smallest = np.unravel_index(dists.argmin(), dists.shape)

					label = temp_lab[smallest[1]]
			labels.append(label)


	return np.array(labels)

def vector_quantization(data, clusters, distance_metric, membership_limit):
	current_clustering = convert_clusters_to_label_representations(data, clusters, membership_limit, distance_metric)
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	features = np.array(list(map(lf, data)))
	labels_, _ = vq(features, [len(features[0]) * [list(np.unique(np.array(current_clustering))).index(cc)] for cc in current_clustering])
	labels_ = labels_.astype(str).tolist()

	unique_labels = np.unique(labels_)

	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, labels_), get_image_indices_for_label(label, data, labels_)) for label in new_unique_labels]

def spectral_clustering(data, clusters, distance_metric, membership_limit, n_clusters):
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))

	spec = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr')
	
	clusters_ = spec.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()
	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]

def agglomerative_clustering(data, clusters, distance_metric, membership_limit, n_clusters):
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))
	
	wrd = AgglomerativeClustering(n_clusters=n_clusters)

	gc.collect()

	clusters_ = wrd.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()
	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	return [cluster(label, [], get_images_for_label(label, data, clusters_.labels_), get_image_indices_for_label(label, data, clusters_.labels_)) for label in new_unique_labels]

