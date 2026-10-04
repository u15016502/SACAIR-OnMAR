import numpy as np
from applications.clustering_composition.utils import *
from applications.clustering_composition.stopping_criteria import * 
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import mutual_info_score, adjusted_rand_score

# https://www.geeksforgeeks.org/clustering-metrics/

def run(chromosome, dataloader, timestep, state=None):
	if state == None:
		clusters = None
		images = None
		labels = None 
		for i, data in enumerate(dataloader, 0):
			images, labels = data
			break 
		n_clusters = len(list(set(labels)))
		images = images.reshape((images.shape[0], images.shape[-1], images.shape[-2], images.shape[-3]))
		data = np.array(get_feature_extraction(chromosome[-1], images))
		clusters = get_cluster_initialization(chromosome[1], data, n_clusters, chromosome[-4], chromosome[0])
		stopped = False
		state = []
	else:
		clusters = state[-1]['clusters']
		images = state[-1]['images']
		labels = state[-1]['labels']
		n_clusters = len(list(set(labels)))
		data = np.array(get_feature_extraction(chromosome[-1], images))

		if chromosome[8] == 0:
			stopped = check_for_changes_in_cluster_assignments(state)
		if chromosome[8] == 1:
			stopped = check_for_changes_in_cluster_variance(state)

	if stopped == False:
		if chromosome[-2] == 0:
			clusters = get_cluster_creation(chromosome[2], data, clusters, n_clusters, chromosome[-4], chromosome[0])
		if chromosome[-2] == 1:
			clusters = get_cluster_addition(chromosome[3], data, clusters, n_clusters, chromosome[-4], chromosome[0])
		if chromosome[-2] == 2:
			clusters = get_cluster_removal(chromosome[4], data, clusters, n_clusters, chromosome[-4], chromosome[0])
		if chromosome[-2] == 3:
			clusters = get_cluster_merging(chromosome[5], data, clusters, n_clusters, chromosome[-4], chromosome[0])
		if chromosome[-2] == 4:
			clusters = get_cluster_splitting(chromosome[6], data, clusters, n_clusters, chromosome[-4], chromosome[0])

	predicted_labels = convert_clusters_to_label_representations(data, clusters, chromosome[-4], chromosome[0])
	fitness = clustering_accuracy(predicted_labels, labels)
		
	a = np.array(predicted_labels)
	b = np.array(labels)
	
	a = a.flatten()
	b = b.flatten()

	labels_pred = a.astype(np.int64)
	labels_true = b.astype(np.int64)

	predicted_labels = convert_labels(labels_pred, labels_true)

	if len(list(set(predicted_labels))) == 1:
		silhouette = -1
	else:
		silhouette = silhouette_score(flatten_or_return(np.array(data), 2), predicted_labels)
	
	if len(list(set(predicted_labels))) == 1:
		db_index = -1
	else:
		db_index = davies_bouldin_score(flatten_or_return(np.array(data), 2), predicted_labels)
	
	if len(list(set(predicted_labels))) == 1:
		ch_index = -1
	else:
		ch_index = calinski_harabasz_score(flatten_or_return(np.array(data), 2), predicted_labels)

	ari = adjusted_rand_score(labels_true, predicted_labels)
	mi = mutual_info_score(labels_true, predicted_labels)

	print(str(timestep) + '\t' + str(fitness) + '\t' + str(silhouette) + '\t'+ str(db_index) + '\t'+ str(ch_index) + '\t'+ str(ari) + '\t'+ str(mi) + '\t' + str(chromosome))

	state.append({
		'clusters': clusters,
		'images': images,
		'labels': labels,
		'data': data
	})

	return fitness, state 