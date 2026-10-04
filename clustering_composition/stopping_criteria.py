import numpy as np

def check_for_changes_in_cluster_assignments(state):

	if len(state) <= 2:
		return False
	else:
		first_checkpoint = state[-2]['clusters']
		second_checkpoint = state[-1]['clusters']

		if len(first_checkpoint) != len(second_checkpoint):
			return False 
		
		checks = [True] * len(first_checkpoint)

		for idx1, c1 in enumerate(first_checkpoint):
			for idx2, c2 in enumerate(second_checkpoint):
				if len(c1.vectors) == len(c2.vectors) and np.sum(np.array(c1.centroid)) == np.sum(np.array(c2.centroid)):
					checks[idx1] = True

		if (False in checks) == False:
			return True
		else:
			return False 

def check_for_changes_in_cluster_variance(state):
	if len(state) <= 1:
		return False

	first_checkpoint = [c_.centroid for c_ in state[-2]['clusters']]
	second_checkpoint = [c_.centroid for c_ in state[-2]['clusters']]

	first_variance = np.var(first_checkpoint)
	second_variance = np.var(second_checkpoint)

	if round(first_variance, 4) == round(second_variance, 4):
		return True
	else:
		return False
