from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import xgboost as xgb
import os 
import numpy as np

class metalearner:  
	def __init__(self, classifier_type):
		self.X = np.array([])
		self.y = np.array([])
		self.classifier_type = classifier_type
		if self.classifier_type == 'NN':
			self.init_nearest_neighbours()
			# https://scikit-learn.org/stable/modules/neighbors.html

		if self.classifier_type == 'RF':
			self.init_random_forest()
			# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

		if self.classifier_type == 'XGB':
			self.init_xgboost()
			# https://www.kaggle.com/code/stuarthallows/using-xgboost-with-scikit-learn

		self.best_for_timestep = []
		self.timestep_checkpoint = []
		self.features = []
		self.fitnesses = []
		self.c = []
	
	def init_nearest_neighbours(self):
		self.yf_learner = KNeighborsRegressor(n_neighbors=5, weights='distance')
		self.cf_learner = KNeighborsRegressor(n_neighbors=5, weights='distance')
		self.yc_learner = KNeighborsRegressor(n_neighbors=5, weights='distance')

	def init_random_forest(self):
		self.yf_learner = RandomForestRegressor(n_estimators=100)
		self.cf_learner = RandomForestRegressor(n_estimators=100)
		self.yc_learner = RandomForestRegressor(n_estimators=100)

	def init_xgboost(self):
		self.yf_learner = xgb.XGBRegressor(tree_method="hist")
		self.cf_learner = xgb.XGBRegressor(tree_method="hist")
		self.yc_learner = xgb.XGBRegressor(tree_method="hist")

	def train_to_predict_y_from_features(self, f, y):
		self.yf_learner.fit(f, y)

	def train_to_predict_c_from_features(self, f, c):
		self.cf_learner.fit(f, c)

	def train_to_predict_y_from_c(self, c, y):
		self.yc_learner.fit(c, y)

	def predict_y(self, c):
		# print(f)
		# print(self.learner.predict([f]))
		return self.yc_learner.predict([c])[0]
	
	def predict_c(self, f):
		# print(f)
		# print(self.learner.predict([f]))
		return self.cf_learner.predict([f])[0]

	def predict_y_from_feats(self, f):
		return self.yf_learner.predict([f])[0]
	