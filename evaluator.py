import numpy as np
from dataset import Dataset
from model import Model


class Evaluator:

	def __init__(self, n_ranks = [1,3,5]):
		"""
		Attribute 'self.n_ranks': For defining the metrics e.g. if n_ranks = [1,5], and the metrics used is RECALL, 
								  then RECALL@1 and RECALL@5 will be used
		Attribute 'self.scores_list': Accumulate/Collect the scores of each of the metrics
		Attribute 'self.count_tuples_tested_on': The total number of tuples that the model has been tested on.

		"""

		self.n_ranks = n_ranks

		self.scores_list = []
		for i, n in enumerate(self.n_ranks):
			metrics_dict = {'PREC@'+str(n): 0, 
							'MAP@'+str(n): 0, 
							'RECALL@'+str(n): 0, 
							'NDCG@'+str(n): 0, }
			self.scores_list.append(metrics_dict)

		self.count_tuples_tested_on = 0

		

	def evaluateList(self, recommendation_df, test_tuple):
		"""
		evaluate the recommendations given by a model

		param 'recommendation_df': is a Dataframe of length n, where each row represents a recommended item. 
									 The 1st row should be the top item that the model would like to recommend.
		param 'test_tuple': is the tuple given to the model for testing purpose.
							Has format of a tuple: <time, userID, itemID, action, data_split_label>

		"""
		if recommendation_df is None: # no recommendation given due to the first tuple given to model being a test tuple
			return

		self.count_tuples_tested_on += 1

		num_of_recommendations = len(recommendation_df)
		binary_hit_ranked_list = np.array( recommendation_df['itemID'] == test_tuple['itemID'] ).astype(int)  # a '1' at the first index in this list would indicate that the 1st item recommended by the model is hit. Same for second index and 2nd item and so on.


		for i, n in enumerate(self.n_ranks):
			n_length_binary_hit_ranked_list = binary_hit_ranked_list[:n]
			precision_at_k_score = self.precision_at_k(n_length_binary_hit_ranked_list, n_length_binary_hit_ranked_list.size)
			average_precision_score = self.average_precision(n_length_binary_hit_ranked_list)
			recall_at_k_score = self.recall_at_k(n_length_binary_hit_ranked_list, num_of_actual_true_labels = 1)
			ndcg_at_k_score = self.ndcg_at_k(n_length_binary_hit_ranked_list)

			self.scores_list[i]['PREC@'+str(n)] += precision_at_k_score
			self.scores_list[i]['MAP@'+str(n)] += average_precision_score
			self.scores_list[i]['RECALL@'+str(n)] += recall_at_k_score
			self.scores_list[i]['NDCG@'+str(n)] += ndcg_at_k_score



		

	def print_results(self):
		
		print("Printing results after testing on ", self.count_tuples_tested_on ," tuples")

		for metrics_dict in self.scores_list:
			for key, value in metrics_dict.items():
				print(key, " : " , value/self.count_tuples_tested_on)


	def precision_at_k(self, binary_hit_ranked_list, k):
		"""
		compute the precision @ k

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.
		param 'k': to compute precision @ k 

		return precision @ k score

		"""
		return np.mean(binary_hit_ranked_list[:k])

	def average_precision(self, binary_hit_ranked_list):
		"""
		compute the average_precision, will award higher scores to recommendation lists that put the hit item at a higher position.

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.

		return average_precision score

		"""
		if np.sum(binary_hit_ranked_list) == 0: # there is no hit in the binary_hit_ranked_list
			return 0

		sum_of_scores = [self.precision_at_k(binary_hit_ranked_list, k) for k in range(1, binary_hit_ranked_list.size + 1 ) if binary_hit_ranked_list[k-1]]
		return np.mean(sum_of_scores)

	def recall_at_k(self, binary_hit_ranked_list, num_of_actual_true_labels = 1):
		"""
		compute the recall @ k 

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.
		Note: The value of k is implied by the length of binary_hit_ranked_list
		param 'num_of_actual_true_labels': The number of True Positives + False Negatives 
													   Value is defaulted to 1 because model is given 1 test tuple at a time.

		return recall @ k score 
		Note: The return value is basically either 0 or 1.

		"""

		return np.sum(binary_hit_ranked_list)/ num_of_actual_true_labels


	def ndcg_at_k(self, binary_hit_ranked_list):
		"""
		Compute the normalized discounted cumulative gain (ndcg). 
		The gain/relevance score is binary i.e. either 0 or 1

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.

		return: ndcg @ k score

		"""
		if np.sum(binary_hit_ranked_list) == 0: # there is no hit in the binary_hit_ranked_list
			return 0

		ideal_order_list = np.asfarray(sorted(binary_hit_ranked_list, reverse=True))

		dcg_ideal = np.sum(ideal_order_list / np.log2(np.arange(2, ideal_order_list.size + 2))) 
		dcg_ranking = np.sum(binary_hit_ranked_list / np.log2(np.arange(2, binary_hit_ranked_list.size + 2)))  #

		return dcg_ranking / dcg_ideal

	def evaluateModel( self, dataset, model, num_of_tuples_to_use = 100, use_cross_validate = True):

		"""

		Evaluate the given model and print the evaluation results.

		param 'dataset' must be a subclass of Dataset and implement the abstract methods
		param 'model' must be a subclass of Model and implment the abstract methods
		param 'use_cross_validate' determines whether to use cross validation (train on train set and not validation set); or not (train on both train and validation sets)

		Return 'self.scores_list' which contains the evaluation results of the model.
		type: a list of dicts

		"""

		if not isinstance(dataset, Dataset):
			raise ValueError('Your dataset object needs to be a subclass of Dataset and implements the abstract methods.')

		if not isinstance(model, Model):
			raise ValueError('Your model needs to be a subclass of Model and implements the abstract methods.')



		STOPPED HERE - To do: how to evaluate valid, test separately!? """
		if use_cross_validate:
			sets_for_training = ['Train']
			sets_for_testing = ['Valid']
		else: 
			sets_for_training = ['Train', 'Valid']
			sets_for_testing = ['Test']
		"""


		"""

		'tuple' is a row of a Dataframe i.e. a Series
		'list_of_recommendations' is a Dataframe of length n, where each row represents a recommended item. The 1st row should be the top item that the model would like to recommend.

		"""

		for iteration_no in range(num_of_tuples_to_use):
			tuple = dataset.getNextTuple()
			if tuple['data_split_label'] == 'Train':
				model.update_model(tuple)
			elif tuple['data_split_label'] == 'Test':
				list_of_recommendations = model.test_model(tuple['userID'])
				self.evaluateList(list_of_recommendations, tuple)

				model.update_model(tuple) # train on the test tuple after the evaluation has been done.
			else:
				raise NotImplementedError

		self.print_results()
		return self.scores_list # return the evaluation results of the model



