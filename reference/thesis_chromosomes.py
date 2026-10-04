from functools import total_ordering
import random 
from copy import deepcopy 
from clustering_composition import run as clustering_composition_run
from neural_network_configuration import run as neural_network_configuration_run
from video_configuration import run as video_configuration_run

@total_ordering
class individual:
	def _init_(self, application):
		self.application = application
		self.fitness = -1
		
		if application == "video_configuration":
			self.chromosome = [
				random.choice([1, 2, 3, 4, 5]), 				# keyframe extraction
				random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10]),	# number of segments
				random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),		# base architecture
				random.choice([1, 2, 3, 4, 5]),					# consensus function
				random.uniform(0.0009, 0.01),					# learning rate
				random.uniform(0.4, 0.65),						# dropout
				random.uniform(1.0,20.0)						# gradient norm clipping
			]

		elif application == "neural_network_configuration":
			num_conv_layers = random.choice([1,2,3,4,5,6])
			num_dense_layers = random.choice([1,2,3,4,5,6])
			self.chromosome = []

			for c in range(num_conv_layers):
				self.chromosome += [
					random.choice([8, 16, 32, 64, 128, 256, 512, 1024, 2048]),	# convolutional filters
					random.choice([0, 1]),										# batch normalisation
					random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),					# activation function
					random.choice([-1, random.uniform(0.0, 1.0)]),				# dropout
					random.choice([0, 2, 4, 6, 8]),								# max pooling 
				]
			
			for d in range(num_dense_layers):
				self.chromosome += [
					random.choice([16, 32, 64, 128, 256, 512, 1024, 2048, 4096]),	# dense nodes
					random.choice([0, 1]),											# batch normalisation
					random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]),						# activation function
					random.choice([-1, random.uniform(0.0, 1.0)]),					# dropout

				]

			self.chromosome += [
				num_conv_layers, 
				num_dense_layers, 
				random.choice([1, 2, 3, 4, 5, 6, 7, 8])]			# optimiser 

		elif application == "clustering_composition":
			self.chromosome = [
				random.choice([0, 1, 2, 3, 4, 5, 6, 7]), # distance metric
				random.choice([0, 1, 2, 3, 4, 5, 6, 7]), # cluster initialization 
				random.choice([0, 1, 2, 3, 4, 5, 6]), # cluster creation
				random.choice([0, 1, 2, 3, 4]), # cluster addition
				random.choice([0, 1, 2, 3, 4, 5]), # cluster removal
				random.choice([0, 1, 2, 3]), # cluster merging
				random.choice([0, 1, 2, 3, 4, 5, 6]), # cluster splitting
				random.choice([0, 1]), # membership limit
				random.choice([0, 1]), # stopping criteria
				random.choice([0, 1, 2, 3, 4]), # step
				random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8]), # feature extraction
			]
		else:
			exit(-1)

	def __eq__(self, other):
		return self.fitness == other.fitness
				
	def __lt__(self, other):
		return self.fitness < other.fitness
	
	def evaluate(self, trainX, trainY, testX, testY, timestep):
		if self.application == "video_configuration":
			video_configuration_run(self.chromosome, trainX, trainY, testX, testY, timestep)
		elif self.application == "neural_network_configuration":
			neural_network_configuration_run(self.chromosome, trainX, trainY, testX, testY, timestep)
		elif self.application == "clustering_composition":
			clustering_composition_run(self.chromosome, trainX, trainY, testX, testY, timestep)
		else:
			exit(-1)