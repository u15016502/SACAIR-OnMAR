"""
Stopping criteria for the clustering composition application.

A stopping criterion decides whether the clustering has converged. When it
reports True the timestep's composition operator (creation, addition,
removal, merging or splitting) is skipped and the existing clusters are scored
as they stand - so a criterion that is wrong in the "stop" direction freezes
the clustering for the rest of the run.

Both criteria as recovered were wrong in exactly that direction:

* ``check_for_changes_in_cluster_assignments`` initialised its match vector to
  ``[True] * n`` and then only ever assigned ``True`` to it, so the ``False in
  checks`` test could never fire and the function returned True for every pair
  of checkpoints with the same number of clusters. The vector now starts at
  ``[False] * n``, which is what the subsequent test expects: a cluster counts
  as unchanged only once a counterpart with the same size and centroid sum has
  actually been found for it.
* ``check_for_changes_in_cluster_variance`` compared ``state[-2]`` with
  ``state[-2]`` - the same checkpoint twice - so the variances were always
  equal and it returned True from the second timestep onward. It now compares
  ``state[-2]`` with ``state[-1]``.

Both take the run's ``state`` list, whose entries are the per-timestep
dictionaries that :func:`clustering_composition.run.run` appends.
"""

import numpy as np


def check_for_changes_in_cluster_assignments(state):
	"""Whether cluster membership has settled.

	Args:
		state: The run's per-timestep state list.

	Returns:
		True if every cluster in the previous checkpoint has a counterpart in
		the latest one with the same size and the same centroid sum.
	"""

	if len(state) <= 2:
		return False
	else:
		first_checkpoint = state[-2]['clusters']
		second_checkpoint = state[-1]['clusters']

		if len(first_checkpoint) != len(second_checkpoint):
			return False

		# One flag per cluster in the earlier checkpoint, each raised only when
		# a matching cluster is found in the later one.
		checks = [False] * len(first_checkpoint)

		for idx1, c1 in enumerate(first_checkpoint):
			for idx2, c2 in enumerate(second_checkpoint):
				if len(c1.vectors) == len(c2.vectors) and np.sum(np.array(c1.centroid)) == np.sum(np.array(c2.centroid)):
					checks[idx1] = True

		if (False in checks) == False:
			return True
		else:
			return False


def check_for_changes_in_cluster_variance(state):
	"""Whether the variance across cluster centroids has settled.

	Args:
		state: The run's per-timestep state list.

	Returns:
		True if the centroid variance is unchanged, to four decimal places,
		between the previous checkpoint and the latest one.
	"""
	if len(state) <= 1:
		return False

	first_checkpoint = [c_.centroid for c_ in state[-2]['clusters']]
	second_checkpoint = [c_.centroid for c_ in state[-1]['clusters']]

	if len(first_checkpoint) == 0 or len(second_checkpoint) == 0:
		return False

	first_variance = np.var(first_checkpoint)
	second_variance = np.var(second_checkpoint)

	if round(first_variance, 4) == round(second_variance, 4):
		return True
	else:
		return False
