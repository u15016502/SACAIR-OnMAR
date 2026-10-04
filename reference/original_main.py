# Python program to demonstrate
# command line arguments


import argparse

from metalearner.metalearner import metalearner
from genetic_algorithm.individual import individual
from dataset import load_image_dataset, load_video_dataset

if __name__ == '__main__':
	# Initialize parser
	parser = argparse.ArgumentParser()

	# Adding optional argument
	parser.add_argument("-a", "--application", choices=["video_configuration", "neural_network_configuration", "clustering_composition"])
	parser.add_argument("-d", "--dataset", choices=["mnist", "fashion-mnist", "cifar10", "cifar100", "mosquito", "fruitsgb", "melanoma", "ucf101", "hmdb51", "lmtd"])
	parser.add_argument("-m", "--meta_learner", choices=["xgb","knn","rf"])
	parser.add_argument("-g", "--num_generations", type=int)
	parser.add_argument("-p", "--population_size", type=int)

	# Read arguments from command line
	args = parser.parse_args()

	if args.application == "video_configuration":
		timesteps = 100
	elif args.application == "neural_network_configuration":
		timesteps = 80
	elif args.application == "clustering_composition":
		timesteps = 60
	else: 
		exit(-1)

	ml = metalearner()

	for t in timesteps:
		if args.application == "video_configuration":
			trainX, trainY, testX, testY = 
		else:
			trainX, trainY, testX, testY = 

		if t == 0:
			pop = [individual(args.application) for i in range(args.population_size)]
		else:
			pop = [individual(args.application) for i in range(args.population_size - 1)] + [best_ind]

		best_ind = max(pop)