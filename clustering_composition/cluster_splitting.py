import time
from scipy.spatial.distance import cdist
import gc
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from applications.clustering_composition.cluster import cluster
import numpy as np
import random
from scipy.spatial.distance import cdist
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

def split_clusters_random_ward_criterion(data, clusters, distance_metric):
	if len(data) == len(clusters) or len(clusters) == 0:
		return clusters

	indices = list([idx for idx, c in enumerate(clusters) if len(c.vectors) > 0])#list(range(0,len(clusters)))
	if len(indices) == 0:
		return clusters

	random.shuffle(indices)
	random_cluster = clusters[indices[0]]

	while len(random_cluster.vectors) <= 1:
		random.shuffle(indices)
		random_cluster = clusters[indices[0]]
		indices.pop(0)
		if len(indices) == 0:
			return clusters
	
	if len(random_cluster.vectors) <= 1:
		return clusters

	wrd = AgglomerativeClustering(n_clusters=2)
	gc.collect()

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, random_cluster.vectors)))

	clusters_ = wrd.fit([flatten_or_return(dataset_instance, 1) for dataset_instance in random_cluster.vectors])
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()

	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	new_clusters = [cluster(label, [], get_images_for_label(label, random_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, random_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
	return [cluster_ for cluster_ in clusters if cluster_.identifier != random_cluster.identifier] + new_clusters

def split_clusters_random_eigenvalues(data, clusters, distance_metric):
	if len(data) == len(clusters) or len(clusters) == 0:
		return clusters

	indices = list([idx for idx, c in enumerate(clusters) if len(c.vectors) > 0])#list(range(0,len(clusters)))
	if len(indices) == 0:
		return clusters

	random.shuffle(indices)
	random_cluster = clusters[indices[0]]

	while len(random_cluster.vectors) <= 1:
		random_cluster = clusters[indices[0]]
		indices.pop(0)
		if len(indices) == 0:
			return clusters	

	if len(random_cluster.vectors) <= 1:
		return clusters

	spec = SpectralClustering(n_clusters=2, assign_labels='cluster_qr')
	
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, random_cluster.vectors)))

	clusters_ = spec.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()
	
	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	new_clusters = [
		cluster(
			label, 
			[], 
			get_images_for_label(label, random_cluster.vectors, clusters_.labels_), 
			get_image_indices_for_label(label, random_cluster.vectors, clusters_.labels_)) 
		for label in new_unique_labels]
	
	return [cluster_ for cluster_ in clusters if cluster_.identifier != random_cluster.identifier] + new_clusters

def split_clusters_with_most_dataset_instances_by_eigenvalues(data, clusters, distance_metric):
	if len(data) == len(clusters) or len(clusters) == 0:
		return clusters

	largest_number_instances = 0
	most_instances_cluster = None

	for c in clusters:
		num_instances = len(c.vectors)

		if num_instances > 1 and num_instances > largest_number_instances:
			largest_number_instances = num_instances
			most_instances_cluster = c

	if most_instances_cluster == None:
		return clusters

	try:
		spec = SpectralClustering(n_clusters=2, assign_labels='cluster_qr')
		gc.collect()
		lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
		data_ = np.array(list(map(lf, most_instances_cluster.vectors)))

		clusters_ = spec.fit(data_)
		clusters_.labels_ = clusters_.labels_.astype(str).tolist()
		unique_labels = np.unique(clusters_.labels_)
		new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

		for idx1, item in enumerate(clusters_.labels_):
			for idx2, ul in enumerate(unique_labels):
				if item == ul:
					clusters_.labels_[idx1] = new_unique_labels[idx2]

		new_clusters = [cluster(label, [], get_images_for_label(label, most_instances_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, most_instances_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
		return [cluster_ for cluster_ in clusters if cluster_.identifier != most_instances_cluster.identifier] + new_clusters
	except:
		return clusters

def split_clusters_with_most_dataset_instances_by_ward_criterion(data, clusters, distance_metric):
	if len(data) == len(clusters) or len(clusters) == 0:
		return clusters

	largest_number_instances = 0
	most_instances_cluster = None

	for c in clusters:
		num_instances = len(c.vectors)

		if num_instances > 1 and num_instances > largest_number_instances:
			largest_number_instances = num_instances
			most_instances_cluster = c

	if most_instances_cluster == None:
		return clusters

	wrd = AgglomerativeClustering(n_clusters=2)

	gc.collect()

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, most_instances_cluster.vectors)))
	
	clusters_ = wrd.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()
	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	new_clusters = [cluster(label, [], get_images_for_label(label, most_instances_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, most_instances_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
	return [cluster_ for cluster_ in clusters if cluster_.identifier != most_instances_cluster.identifier] + new_clusters

def split_clusters_by_ward_criterion(data, clusters, distance_metric):
	if len(data) == len(clusters) or len(clusters) == 0:
		return clusters

	largest_distance_val = []
	largest_distance_val_cluster = None

	if len(clusters) == 1 and len(clusters[0].vectors) > 1:
		largest_distance_val_cluster = clusters[0]
	else:
		for cluster_ in clusters:
			distances = []
			if len(cluster_.vectors) > 1:
				flattened_vectors = np.array([vec.flatten() for vec in cluster_.vectors])
				distances = [(vec, idx) for idx, vec in enumerate(flattened_vectors[0:500]) if idx < len(flattened_vectors) - 1]

				lf = lambda tup: apply_distance_metric(distance_metric, tup[0].reshape(1, -1), flattened_vectors[tup[1] + 1].reshape(1, -1)) 
				distances = np.array(list(map(lf, distances)))
	

				if len(largest_distance_val) == 0 or np.average(distances) > np.average(largest_distance_val):
					largest_distance_val = distances
					largest_distance_val_cluster = cluster_

	if largest_distance_val_cluster == None:
		return clusters

	gc.collect()

	wrd = AgglomerativeClustering(n_clusters=2)

	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	data_ = np.array(list(map(lf, largest_distance_val_cluster.vectors)))

	clusters_ = wrd.fit(data_)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()
	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	new_clusters = [cluster(label, [], get_images_for_label(label, largest_distance_val_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, largest_distance_val_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
	return [cluster_ for cluster_ in clusters if cluster_.identifier != largest_distance_val_cluster.identifier] + new_clusters

def split_clusters_by_eigenvalues(data, clusters, distance_metric):
	if len(data) == len(clusters) or len(clusters) <= 1:
		return clusters

	largest_distance_val = -1
	largest_distance_val_cluster = None

	if len(clusters) == 1 and len(clusters[0].vectors) > 1:
		largest_distance_val_cluster = clusters[0]
	else:
		for cluster_ in clusters:
			distances = []
			if len(cluster_.vectors) > 1:
				lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
				points = np.array(list(map(lf, cluster_.vectors)))
				
				distances = cdist(points, points, metric=get_distance_metric(distance_metric).replace('manhattan', 'cityblock'))
					
				if largest_distance_val_cluster == None or np.average(distances) > np.average(largest_distance_val):
					largest_distance_val = distances
					largest_distance_val_cluster = cluster_

	if largest_distance_val_cluster == None:
		return clusters

	spec = SpectralClustering(n_clusters=2, assign_labels='cluster_qr')
	lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
	points = np.array(list(map(lf, largest_distance_val_cluster.vectors)))

	clusters_ = spec.fit(points)
	clusters_.labels_ = clusters_.labels_.astype(str).tolist()
	unique_labels = np.unique(clusters_.labels_)
	new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

	for idx1, item in enumerate(clusters_.labels_):
		for idx2, ul in enumerate(unique_labels):
			if item == ul:
				clusters_.labels_[idx1] = new_unique_labels[idx2]

	new_clusters = [cluster(label, [], get_images_for_label(label, largest_distance_val_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, largest_distance_val_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
	return [cluster_ for cluster_ in clusters if cluster_.identifier != largest_distance_val_cluster.identifier] + new_clusters

def split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, linkage, criterion):
	if len(data) == len(clusters) or len(clusters) == 0:
		return clusters

	if criterion == 'largest':

		largest_distance_val = []
		largest_distance_val_cluster = None

		if len(clusters) == 1 and len(clusters[0].vectors) > 1:
			largest_distance_val_cluster = clusters[0]
		else:
			for cluster_ in clusters:
				distances = []
				if len(cluster_.vectors) > 1:
					flattened_vectors = np.array([vec.flatten() for vec in cluster_.vectors])
					distances = [apply_distance_metric(distance_metric, vec.reshape(1, -1), flattened_vectors[idx + 1].reshape(1, -1)) for idx, vec in enumerate(flattened_vectors[0:500]) if idx < len(flattened_vectors) - 1]

					if len(largest_distance_val) == 0 or np.average(distances) > np.average(largest_distance_val):
						largest_distance_val = distances
						largest_distance_val_cluster = cluster_

		if largest_distance_val_cluster == None:
			return clusters

		gc.collect()

		wrd = AgglomerativeClustering(n_clusters=2, linkage=linkage)
		lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
		data_ = np.array(list(map(lf, largest_distance_val_cluster.vectors)))

		clusters_ = wrd.fit(data_)
		clusters_.labels_ = clusters_.labels_.astype(str).tolist()
		unique_labels = np.unique(clusters_.labels_)
		new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

		for idx1, item in enumerate(clusters_.labels_):
			for idx2, ul in enumerate(unique_labels):
				if item == ul:
					clusters_.labels_[idx1] = new_unique_labels[idx2]

		new_clusters = [cluster(label, [], get_images_for_label(label, largest_distance_val_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, largest_distance_val_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
		return [cluster_ for cluster_ in clusters if cluster_.identifier != largest_distance_val_cluster.identifier] + new_clusters

	if criterion == 'most':

		largest_number_instances = 0
		most_instances_cluster = None

		for c in clusters:
			num_instances = len(c.vectors)

			if num_instances > 1 and num_instances > largest_number_instances:
				largest_number_instances = num_instances
				most_instances_cluster = c

		if most_instances_cluster == None:
			return clusters

		wrd = AgglomerativeClustering(n_clusters=2, linkage=linkage)

		gc.collect()

		lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
		data_ = np.array(list(map(lf, most_instances_cluster.vectors)))

		clusters_ = wrd.fit(data_)
		clusters_.labels_ = clusters_.labels_.astype(str).tolist()
		unique_labels = np.unique(clusters_.labels_)
		new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

		for idx1, item in enumerate(clusters_.labels_):
			for idx2, ul in enumerate(unique_labels):
				if item == ul:
					clusters_.labels_[idx1] = new_unique_labels[idx2]

		new_clusters = [cluster(label, [], get_images_for_label(label, most_instances_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, most_instances_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
		return [cluster_ for cluster_ in clusters if cluster_.identifier != most_instances_cluster.identifier] + new_clusters


	if criterion == 'random':
		indices = list([idx for idx, c in enumerate(clusters) if len(c.vectors) > 0])#list(range(0,len(clusters)))
		if len(indices) == 0:
			return clusters

		random.shuffle(indices)
		random_cluster = clusters[indices[0]]

		while len(random_cluster.vectors) <= 1:
			random.shuffle(indices)
			random_cluster = clusters[indices[0]]
			indices.pop(0)
			if len(indices) == 0:
				return clusters
		
		if len(random_cluster.vectors) <= 1:
			return clusters

		wrd = AgglomerativeClustering(n_clusters=2, linkage=linkage)
		gc.collect()

		lf = lambda dataset_instance: flatten_or_return(dataset_instance, 1)
		data_ = np.array(list(map(lf, random_cluster.vectors)))

		clusters_ = wrd.fit(data_)
		clusters_.labels_ = clusters_.labels_.astype(str).tolist()
		unique_labels = np.unique(clusters_.labels_)
		new_unique_labels = [str(time.time()).replace('.','') for ul in unique_labels]

		for idx1, item in enumerate(clusters_.labels_):
			for idx2, ul in enumerate(unique_labels):
				if item == ul:
					clusters_.labels_[idx1] = new_unique_labels[idx2]

		new_clusters = [cluster(label, [], get_images_for_label(label, random_cluster.vectors, clusters_.labels_), get_image_indices_for_label(label, random_cluster.vectors, clusters_.labels_)) for label in new_unique_labels]
		return [cluster_ for cluster_ in clusters if cluster_.identifier != random_cluster.identifier] + new_clusters
