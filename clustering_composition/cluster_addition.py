from collections import Counter
import time
from scipy.spatial.distance import cdist
import gc
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from applications.clustering_composition.cluster import cluster
import numpy as np
from scipy.spatial.distance import cdist
from dataset.utils import get_images_for_label, get_image_indices_for_label

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

def get_neighbourhood_for_dataset_instance(dataset_instance, data, clustering, nearest_neighbours):
	dist, nn_indices = nearest_neighbours.kneighbors([flatten_or_return(dataset_instance,1)])
	neighbours = []

	for idx in nn_indices[0]:
		if clustering[idx] != -1:
			neighbours.append(clustering[idx])

	return neighbours

def add_to_cluster_using_neighbourhood(data, clusters, membership_limit, distance_metric):
	if len(clusters) <= 1:
		return clusters

	new_clustering = []
	current_clustering = convert_clusters_to_label_representations(data, clusters, membership_limit, distance_metric)

	if len(clusters) <= 1 and str(list(np.unique(np.array(current_clustering)))[0]) == '-1':
		return clusters

	unique_labels = list(np.unique([c.identifier for c in clusters]))
	unique_labels_ = [str(time.time()).replace('.','') for ul in unique_labels]
	unassigned = str(time.time()).replace('.','')
	nearest_neighbours = NearestNeighbors(n_neighbors=10, metric=get_distance_metric(distance_metric))
	gc.collect()
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, data)))

	nearest_neighbours.fit(data_)

	for idx, dataset_instance in enumerate(data_):
		neighbourhood = get_neighbourhood_for_dataset_instance(dataset_instance, data_, current_clustering, nearest_neighbours)

		if len(neighbourhood) > 1:
			occurence_count = Counter(neighbourhood)
			most_common_identifier = occurence_count.most_common(1)[0][0]
			try:
				new_clustering.append(unique_labels_[unique_labels.index(most_common_identifier)])
			except:
				new_clustering.append('-1')
		else:
			new_clustering.append(unassigned)

	unique_labels = np.unique(new_clustering)

	new_clusters = [cluster(label, [], get_images_for_label(label, data, new_clustering), get_image_indices_for_label(label, data, new_clustering)) for label in unique_labels]

	if membership_limit == 0:
		return new_clusters
	else:
		return new_clusters + clusters

def add_to_cluster_using_centroids_mp(tup):
	data = tup[0]
	clusters = tup[1]
	membership_limit = tup[2]
	distance_metric = tup[3]

	return add_to_cluster_using_centroids(data, clusters, membership_limit, distance_metric)

def add_to_cluster_using_centroids(data, clusters, membership_limit, distance_metric):
	new_clustering = []

	if len(clusters) == 0:
		return []
	elif len(clusters) == 1:
		new_clustering = [clusters[0].identifier for dataset_instance in data]
		new_clusters = [cluster(label, [], get_images_for_label(label, data, new_clustering), get_image_indices_for_label(label, data, new_clustering)) for label in np.unique(new_clustering)]
		return new_clusters
	else:
		current_clustering = convert_clusters_to_label_representations(data, clusters, membership_limit, distance_metric)
		unique_labels = list(np.unique([c.identifier for c in clusters]))
		unique_labels_ = [str(time.time()).replace('.','') for ul in unique_labels]

		for idx, dataset_instance in enumerate(data):

			closest_cluster = None
			closest_distance = None
			
			for cluster_ in clusters:
				centroid = cluster_.centroid

				dis = apply_distance_metric(distance_metric, flatten_or_return(dataset_instance, 2), flatten_or_return(centroid, 2))

				if closest_cluster == None or (np.average(dis) < np.average(closest_distance)):
					closest_cluster = cluster_
					closest_distance = dis

			new_clustering.append(unique_labels_[unique_labels.index(closest_cluster.identifier)])

		unique_labels = np.unique(new_clustering)
		new_clusters = [cluster(label, [], get_images_for_label(label, data, new_clustering), get_image_indices_for_label(label, data, new_clustering)) for label in unique_labels]

		if membership_limit == 0:
			return new_clusters
		else:
			return new_clusters + clusters

def add_to_cluster_using_gaussian_distribution(data, clusters, membership_limit, distance_metric, random_seed):
	if len(clusters) == 0:
		gm = GaussianMixture( n_init=1, covariance_type='spherical')
	elif len(clusters) == len(data):
		return clusters
	else:
		gm = GaussianMixture( n_init=1, covariance_type='spherical')

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	_data = np.array(list(map(lf, data)))
	clusters_ = gm.fit_predict(_data)
	clusters_ = clusters_.astype(str).tolist()

	unique_labels = np.unique(clusters_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_[idx1] = new_unique_labels[idx2]

	new_clustering = []

	for idx, dataset_instance in enumerate(data):
		new_clustering.append(clusters_[idx])

	unique_labels = np.unique(new_clustering)

	new_clusters = [cluster(label, [], get_images_for_label(label, data, new_clustering), get_image_indices_for_label(label, data, new_clustering)) for label in new_unique_labels]

	if membership_limit == 0:
		return new_clusters
	else:
		return new_clusters + clusters

def add_to_cluster_ensembled(data, clusters, membership_limit, distance_metric, random_seed):
	if len(clusters) == 0:
		return []

	centroid_clusters = add_to_cluster_using_centroids(data, clusters, membership_limit, distance_metric)
	gaussian_clusters = add_to_cluster_using_gaussian_distribution(data, clusters, membership_limit, distance_metric,random_seed=random_seed)
	unique_labels = list(np.unique([cc.identifier for cc in centroid_clusters])) + list(np.unique([gc.identifier for gc in gaussian_clusters]))
	unique_labels_ = [str(time.time()).replace('.','') for ul in np.unique([cc.identifier for cc in centroid_clusters])] + [str(time.time()).replace('.','') for ul in np.unique([gc.identifier for gc in gaussian_clusters])]

	current_clustering = convert_clusters_to_label_representations(data, clusters, membership_limit, distance_metric)
	combined_clusters = centroid_clusters + gaussian_clusters
	new_clustering = []

	for idx, dataset_instance in enumerate(data):
		closest_cluster = None
		closest_distance = None
			
		for cluster_ in combined_clusters:
			centroid = cluster_.centroid

			dis = apply_distance_metric(distance_metric, flatten_or_return(dataset_instance,2), flatten_or_return(centroid,2))

			if closest_cluster == None or (np.average(dis) < np.average(closest_distance)):
				closest_cluster = cluster_
				closest_distance = dis
		
		new_clustering.append(unique_labels_[unique_labels.index(closest_cluster.identifier)])
			
	unique_labels = np.unique(new_clustering)
	new_clusters = [cluster(label, [], get_images_for_label(label, data, new_clustering), get_image_indices_for_label(label, data, new_clustering)) for label in unique_labels]
		
	if membership_limit == 0:
		return new_clusters
	else:
		return new_clusters + clusters
