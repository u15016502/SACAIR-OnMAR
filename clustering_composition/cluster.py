import time
import numpy as np

class cluster:

	def __init__(self, identifier=None, centroid=[], vectors=[], vector_indices=[]):
		if identifier == None:
			identifier = str(time.time())
			identifier = identifier.replace('.','')
			self.identifier = identifier
		else:
			self.identifier = identifier

		self.centroid = None
		self.vectors = vectors
		self.vector_indices = vector_indices

		if len(centroid) <= 0:
			self.set_centroid()
		else:
			self.centroid = centroid

	def set_identifier(self, identifier):
		self.identifier = identifier

	def set_centroid(self):
		self.centroid = np.sum(np.array(self.vectors), axis=0) / len(self.vectors)

	def check_for_vector(self, vector_ind):
		if vector_ind in self.vector_indices:
			return True
		else:
			return False

	def to_string(self):
		print('=============' + str(self.identifier) 
		+ '\t' +  str(len(self.centroid)) 
		+ '\t' +  str(len(self.vectors)) 
		+ '\t' + str(self.vector_indices) + '\t' +  '=============')
