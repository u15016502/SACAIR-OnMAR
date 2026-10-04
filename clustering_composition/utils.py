import numpy as np
from sklearn.metrics import accuracy_score
from applications.clustering_composition.feature_extraction import *
from applications.clustering_composition.cluster import cluster
from applications.clustering_composition.cluster_initialization import *
from applications.clustering_composition.cluster_creation import * 
from applications.clustering_composition.cluster_addition import * 
from applications.clustering_composition.cluster_removal import * 
from applications.clustering_composition.cluster_merging import * 
from applications.clustering_composition.cluster_splitting import * 

def get_feature_extraction(feature_extraction, images):
	
	if feature_extraction == 0:
		return apply_resnet50(images)
	if feature_extraction == 1:
		return apply_inceptionv3(images)
	if feature_extraction == 2:
		return apply_densenet121(images) 
	if feature_extraction == 3:
		return apply_xception(images)
	if feature_extraction == 4:
		return apply_vgg16(images)
	if feature_extraction == 5:
		return apply_pca(images, -1, 1024)
	if feature_extraction == 6:
		return apply_sift(images)
	if feature_extraction == 7:
		return apply_tsne(images, -1, 1024, 100, 10, 59)
	if feature_extraction == 8:
		return apply_brief(images)
	if feature_extraction == 9:
		data = np.array(images).reshape(len(images),-1)
		data = data.astype(float) / 255.
		return data

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
	elif len(vec.shape) == 4 and expected_shape == 1:
		return vec.flatten()
	elif len(vec.shape) == 4 and expected_shape == 2:
		return np.array([v.flatten() for v in vec])
	elif len(vec.shape) == 1 and expected_shape == 2:
		return np.array(vec).reshape(1, -1)
	else:
		return vec.flatten()

def infer_cluster_labels(model, actual_labels):
	inferred_labels = {}

	for i in list(set(model)):   
		labels = []
		index = np.where(model == i)
		labels.append(actual_labels[index])

		if len(labels[0]) == 1:
			counts = np.bincount(labels[0])
		else:
			counts = np.bincount(np.squeeze(labels))

		try:
			if np.argmax(counts) in inferred_labels:
				inferred_labels[np.argmax(counts)].append(i)
			else:
				inferred_labels[np.argmax(counts)] = [i]
		except:
			counts = None
		
	return inferred_labels  

def infer_data_labels(X_labels, cluster_labels):
	"""
	Determines label for each array, depending on the cluster it has been assigned to.
	returns: predicted labels for each array
	"""
	
	# empty array of len(X)
	predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)
	
	for i, cluster in enumerate(X_labels):
		for key, value in cluster_labels.items():
			if cluster in value:
				predicted_labels[i] = key
				
	return predicted_labels

def convert_labels(labels_pred, labels_true):
	all_labels = list(set(labels_pred))
	labels_pred = np.array([all_labels.index(lp) for lp in labels_pred])	
	cluster_labels = infer_cluster_labels(labels_pred, labels_true)
	predicted_labels = infer_data_labels(labels_pred, cluster_labels)
	return predicted_labels
		
def clustering_accuracy(a, b):
	a = np.array(a)
	b = np.array(b)
	a = a.flatten()
	b = b.flatten()

	if len(list(set(list(a)))) == len(a) or len(list(set(list(a)))) + 1 == len(a) or len(list(set(list(a)))) + 2 == len(a):
		return -1.0

	labels_pred = a.astype(np.int64)
	labels_true = b.astype(np.int64)

	predicted_labels = convert_labels(labels_pred, labels_true)
		
	return accuracy_score(labels_true, predicted_labels)

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

def get_cluster_initialization(cluster_initialization, data, n_clusters, membership_limit, distance_metric):

	if cluster_initialization == 0:
		return []
	if cluster_initialization == 1:
		return [cluster(identifier=0, vectors=data, vector_indices=range(0,len(data)))]
	if cluster_initialization == 2:
		clusters = initialize_centroids_kmeans_plus_plus_sillouette_score(data, n_clusters)
		clusters = add_to_cluster_using_centroids(data, clusters, membership_limit, distance_metric)
		return clusters
	if cluster_initialization == 3:
		clusters = initialize_centroids_kmeans_plus_plus_sillouette_score(data, n_clusters)
		clusters = add_to_cluster_using_centroids(data, clusters, membership_limit, distance_metric)
		return clusters
	if cluster_initialization == 4:
		return initialize_centroids_half_half(data)
	if cluster_initialization == 5:
		return initialize_centroids_birch(data)
	if cluster_initialization == 6:
		return initialize_centroids_optics(data, distance_metric)
	if cluster_initialization == 7:
		return initialize_centroids_bisecting_kmeans(data, n_clusters)

def get_cluster_creation(cluster_creation, data, clusters, n_clusters, membership_limit, distance_metric):
	if cluster_creation == 0:
		return minibatch(data, clusters, distance_metric, membership_limit, n_clusters)
	if cluster_creation == 1:
		return mean_shift(data, clusters, membership_limit, n_clusters)
	if cluster_creation == 2:
		return dbscan(data, clusters, membership_limit, distance_metric, n_clusters)
	if cluster_creation == 3:
		return exemplar(data, clusters, membership_limit)
	if cluster_creation == 4:
		return kmediods(data, clusters, distance_metric, membership_limit, n_clusters)
	if cluster_creation == 5:
		return vector_quantization(data, clusters, distance_metric, membership_limit) 
	if cluster_creation == 6:	
		return spectral_clustering(data, clusters, distance_metric, membership_limit, n_clusters)
	if cluster_creation == 7:	
		return agglomerative_clustering(data, clusters, distance_metric, membership_limit, n_clusters)

def get_cluster_addition(cluster_addition, data, clusters, n_clusters, membership_limit, distance_metric):
	
	if cluster_addition == 0:
		return add_to_cluster_using_centroids(data, clusters, membership_limit, distance_metric)
	if cluster_addition == 1:
		return add_to_cluster_using_gaussian_distribution(data, clusters, membership_limit, distance_metric,1)
	if cluster_addition == 2:
		return add_to_cluster_ensembled(data, clusters, membership_limit, distance_metric, 1)
	if cluster_addition == 3:
		return add_to_cluster_using_neighbourhood(data, clusters, membership_limit, distance_metric)
	if cluster_addition == 4:
		return markov_clustering(data, clusters, distance_metric, membership_limit)

def get_cluster_removal(cluster_removal, data, clusters, n_clusters, membership_limit, distance_metric):

	if cluster_removal == 0:
		return clusters
	if cluster_removal == 1:
		return remove_overlapping_clusters(data, clusters)
	if cluster_removal == 2:
		return remove_cluster_largest_inter_distance(data, clusters, distance_metric)
	if cluster_removal == 3:
		return remove_cluster_least_dataset_instances(data, clusters, distance_metric)
	if cluster_removal == 4:
		return remove_cluster_random(data, clusters, distance_metric)
	if cluster_removal == 5:
		return remove_cluster_most_dataset_instances(data, clusters, distance_metric)

def get_cluster_merging(cluster_merging, data, clusters, n_clusters, membership_limit, distance_metric):
	
	if cluster_merging == 0:
		return clusters
	if cluster_merging == 1:
		return merge_clusters_by_distance(data, clusters, distance_metric)
	if cluster_merging == 2:
		return merge_clusters_with_least_dataset_instances(data, clusters, distance_metric)
	if cluster_merging == 3:
		return merge_clusters_random(data, clusters, distance_metric)

def get_cluster_splitting(cluster_splitting, data, clusters, n_clusters, membership_limit, distance_metric):

	if cluster_splitting == 0:
		return clusters
	if cluster_splitting == 1:
		return split_clusters_by_ward_criterion(data, clusters, distance_metric)
	if cluster_splitting == 2:
		return split_clusters_by_eigenvalues(data, clusters, distance_metric)
	if cluster_splitting == 3:
		return split_clusters_with_most_dataset_instances_by_ward_criterion(data, clusters, distance_metric)
	if cluster_splitting == 4:
		return split_clusters_with_most_dataset_instances_by_eigenvalues(data, clusters, distance_metric)
	if cluster_splitting == 5:
		return split_clusters_random_ward_criterion(data, clusters, distance_metric)
	if cluster_splitting == 6:
		return split_clusters_random_eigenvalues(data, clusters, distance_metric)
	if cluster_splitting == 7:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'single', 'largest')
	if cluster_splitting == 8:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'single', 'most')
	if cluster_splitting == 9:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'single', 'random')
	if cluster_splitting == 10:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'average', 'largest')
	if cluster_splitting == 11:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'average', 'most')
	if cluster_splitting == 12:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'average', 'random')
	if cluster_splitting == 13:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'complete', 'largest')
	if cluster_splitting == 14:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'complete', 'most')
	if cluster_splitting == 15:
		return split_clusters_by_criterion_using_agglomeration(data, clusters, distance_metric, 'complete', 'random')

def get_stopping_criteria(stopping_criteria, data, clusters, history):
	if stopping_criteria == 0:
		return check_for_changes_in_cluster_assignments(data, clusters, history)
	if stopping_criteria == 1:
		return check_for_changes_in_cluster_variance(data, clusters, history)


def check_for_changes_in_cluster_assignments(data, clusters, history):
	has_changed = False

	if len(history) <= 5:
		return has_changed
	else:
		first_checkpoint = history[len(history) - 11]
		second_checkpoint = history[len(history) - 1]

		if len(first_checkpoint) != len(second_checkpoint):
			return False
		else:
			for idx, c_ in enumerate(first_checkpoint):
				if len(c_.vectors) != len(second_checkpoint[idx].vectors):
					return False
				if np.sum(np.array(c_.centroid)) != np.sum(np.array(second_checkpoint[idx].centroid)):
					return False

			return True
				

def check_for_changes_in_cluster_variance(data, clusters, history):
	if len(history) <= 1:
		return False
	
	first_checkpoint = [c_.centroid for c_ in history[len(history) - 2]]
	second_checkpoint = [c_.centroid for c_ in history[len(history) - 1]]

	first_variance = np.var(first_checkpoint)
	second_variance = np.var(second_checkpoint)

	if first_variance != second_variance:
		return False
	else:
		return True

def calculate_distance_from_centroids(data, clusters, membership_limit, distance_metric):
	distances = []

	if membership_limit == 0:
		for idx, dataset_instance in enumerate(data):
			dis = -1
			label = -1
			if clusters != None:
				for c_ in clusters:
					if c_.check_for_vector(idx) == True:
						dis = apply_distance_metric(distance_metric, flatten_or_return(c_.centroid, 2), flatten_or_return(dataset_instance, 2))
						break

			distances.append(dis)
	else:
		for idx, dataset_instance in enumerate(data):
			label = -1
			temp_lab = []
			temp_dist = []
			smallest = -1

			for c_ in clusters:
				if c_.check_for_vector(idx) == True:
					temp_lab.append(c_.identifier)
					temp_dist.append(flatten_or_return(c_.centroid, 1))
				
			if len(temp_lab) > 0:
				dists = cdist([flatten_or_return(dataset_instance, 1)], temp_dist, metric=get_distance_metric(distance_metric).replace('manhattan', 'cityblock'))
				dists[dists == 0] = dists.max()
				smallest = np.unravel_index(dists.argmin(), dists.shape)

			distances.append(smallest)

	return np.array(distances)


