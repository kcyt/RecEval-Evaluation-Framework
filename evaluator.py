import numpy as np
import pandas as pd
from datasets.dataset import Dataset
from models.model import Model


class Evaluator:

	def __init__(self, n_ranks = [1,3,5]):
		"""
		Attribute 'self.n_ranks': For defining the metrics e.g. if n_ranks = [1,5], and the metrics used is RECALL, 
								  then RECALL@1 and RECALL@5 will be used
		type: list

		Attribute 'self.scores_list': Accumulate/Collect the scores of each of the metrics
		type: list of dicts

		Attribute 'self.validation_scores_list': Same purpose as self.scores_list, but used for validation instead of testing.
		type: list of dicts

		Attribute 'self.count_tuples_evaluated_on': The total number of tuples that the model has been evaluated on.
		type: int

		"""

		self.n_ranks = n_ranks

		self.scores_list = []
		for i, n in enumerate(self.n_ranks):
			metrics_dict = {'PREC@'+str(n): 0, 
							'MAP@'+str(n): 0, 
							'RECALL@'+str(n): 0, 
							'NDCG@'+str(n): 0, }
			self.scores_list.append(metrics_dict)
		self.validation_scores_list = self.scores_list.copy()

		self.count_tuples_evaluated_on = 0

		

	def evaluateList(self, recommendation_df, test_tuple):
		"""
		param 'recommendation_df': is a Dataframe of length n, where each row represents a recommended item. 
									 The 1st row should be the top item that the model would like to recommend.
		type: pd.DataFrame

		param 'test_tuple': is the tuple given to the model for testing purpose.
							Has format of a tuple: <time, userID, itemID, action, data_split_label>
		type: pd.Series

		Evaluate the recommendations given by a model.

		No return object.

		"""
		if recommendation_df is None: 
			raise ValueError("recommendation_df given to evaluateList() is empty!")
			

		self.count_tuples_evaluated_on += 1

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
		"""

		print the evaluation metrics scores.

		No return object.

		"""
		
		print("Printing results after testing on ", self.count_tuples_evaluated_on ," tuples")

		for metrics_dict in self.scores_list:
			for key, value in metrics_dict.items():
				print(key, " : " , value/self.count_tuples_evaluated_on)


	def precision_at_k(self, binary_hit_ranked_list, k):
		"""

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.
		type: np.array 

		param 'k': to compute precision @ k 
		type: int

		Compute and return the precision @ k score

		return type: float/double

		"""
		return np.mean(binary_hit_ranked_list[:k])

	def average_precision(self, binary_hit_ranked_list):
		"""

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.
		type: np.array 

		Compute and return the average_precision
		Note: this function will award higher scores to recommendation lists that put the hit item at a higher position.

		return type: float/double

		"""
		if np.sum(binary_hit_ranked_list) == 0: # there is no hit in the binary_hit_ranked_list
			return 0

		sum_of_scores = [self.precision_at_k(binary_hit_ranked_list, k) for k in range(1, binary_hit_ranked_list.size + 1 ) if binary_hit_ranked_list[k-1]]
		return np.mean(sum_of_scores)

	def recall_at_k(self, binary_hit_ranked_list, num_of_actual_true_labels = 1):
		"""

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.
		Note: The value of k is implied by the length of binary_hit_ranked_list
		type: np.array 
		
		param 'num_of_actual_true_labels': The number of True Positives + False Negatives 
													   Value is defaulted to 1 because model is given 1 test tuple at a time.
		type: int

		Compute and return the recall @ k score 
		Note: The return value is basically either 0 or 1.

		return type: float/double

		"""

		return np.sum(binary_hit_ranked_list)/ num_of_actual_true_labels


	def ndcg_at_k(self, binary_hit_ranked_list):
		"""
		Compute the normalized discounted cumulative gain (ndcg). 
		The gain/relevance score is binary i.e. either 0 or 1

		param 'binary_hit_ranked_list':  A numpy array whose index that indicates whether there is a hit in that position or not.
		type: np.array 

		Compute the normalized discounted cumulative gain (ndcg) @ k. Return the ndcg @ k score.
		The gain/relevance score is binary i.e. either 0 or 1. 

		return type: float/double

		"""
		if np.sum(binary_hit_ranked_list) == 0: # there is no hit in the binary_hit_ranked_list
			return 0

		ideal_order_list = np.asfarray(sorted(binary_hit_ranked_list, reverse=True))

		dcg_ideal = np.sum(ideal_order_list / np.log2(np.arange(2, ideal_order_list.size + 2))) 
		dcg_ranking = np.sum(binary_hit_ranked_list / np.log2(np.arange(2, binary_hit_ranked_list.size + 2)))  

		return dcg_ranking / dcg_ideal

	def evaluate_validation_set_in_batch(self, recommendation_df, validation_set_df ):

		"""
		param 'recommendation_df': A dataframe containing the recommendations for each user. 
		The dataframe will have 3 columns: 1. 'userID' 2. 'itemID_list', and 3.'action_list'
		The 'itemID_list' column contains a list of recommended itemID for each user.
		The 'action_list' column contains a list of recommended action for each user. Each action must be an enum defined in a subclass of dataset.Action 
		type: pd.DataFrame

		param 'validation_set_df': A dataframe containing the validation tuples from the dataset.
		type: pd.DataFrame

		Evaluate the model using the Validation set and print out the scores. 
		This function returns self.validation_scores_list (a list of dicts) containing the scores.

		return type: a list of dicts.

		"""

		user_grouped_validation_set_df = validation_set_df.groupby('userID')['itemID'].apply(list).reset_index()
		user_grouped_validation_set_df.rename(columns = {'itemID':'actual_itemID_list'} ,inplace=True)
		user_grouped_validation_set_df['actual_action_list'] = validation_set_df.groupby('userID')['action'].apply(list)

		# left outer join to merge 'user_grouped_validation_set_df' and 'recommendation_df'
		merged_user_grouped_df = user_grouped_validation_set_df.merge(recommendation_df, 
               how = 'left',  
               left_on = 'userID',     
               right_on = 'userID')  

		max_n = np.max(self.n_ranks)

		# define an inner function to compute binary_hit_ranked_list 
		def compute_max_binary_hit_ranked_list(row):
			predicted_itemID_list = row['itemID_list']
			predicted_itemID_list = predicted_itemID_list[:max_n]
			actual_itemID_list = row['actual_itemID_list']
			intersected_itemIDs = list(set(predicted_itemID_list).intersection(set(actual_itemID_list) ) )
			max_binary_hit_ranked_list = np.zeros(max_n)
			for itemID in intersected_itemIDs:
				max_binary_hit_ranked_list[ predicted_itemID_list.index(itemID)] = 1
			return max_binary_hit_ranked_list

		merged_user_grouped_df['max_binary_hit_ranked_list'] = merged_user_grouped_df.apply(func=compute_max_binary_hit_ranked_list , axis = 1)

		# define an inner function to compute overall_recall_at_k_score
		def compute_overall_recall_at_k_score(row):
			binary_hit_ranked_list = row['current_binary_hit_ranked_list']
			num_of_actual_true_labels = len(row['actual_itemID_list'])
			return np.sum(binary_hit_ranked_list)/ num_of_actual_true_labels


		for i,n in enumerate(self.n_ranks):
			merged_user_grouped_df['current_binary_hit_ranked_list'] = merged_user_grouped_df['max_binary_hit_ranked_list'].apply(lambda x: x[:n])
			overall_precision_at_k_score =  merged_user_grouped_df['current_binary_hit_ranked_list'].apply(func = self.precision_at_k, args=(n,)).mean()
			overall_average_precision_score = merged_user_grouped_df['current_binary_hit_ranked_list'].apply(func = self.average_precision).mean()
			overall_recall_at_k_score = merged_user_grouped_df.apply(func=compute_overall_recall_at_k_score, axis=1 ).mean()   
			overall_ndcg_at_k_score = merged_user_grouped_df['current_binary_hit_ranked_list'].apply(func = self.ndcg_at_k).mean() 


			self.validation_scores_list[i]['PREC@'+str(n)] = overall_precision_at_k_score
			self.validation_scores_list[i]['MAP@'+str(n)] = overall_average_precision_score
			self.validation_scores_list[i]['RECALL@'+str(n)] = overall_recall_at_k_score
			self.validation_scores_list[i]['NDCG@'+str(n)] = overall_ndcg_at_k_score

			print('PREC@'+str(n)+': ', overall_precision_at_k_score)
			print('MAP@'+str(n)+': ', overall_average_precision_score)
			print('RECALL@'+str(n)+': ', overall_recall_at_k_score)
			print('NDCG@'+str(n)+': ', overall_ndcg_at_k_score)

		return self.validation_scores_list

		


	def evaluate_model_in_stream( self, dataset, model, mode = 'Testing', scheme = [0.1,0.9]):

		"""

		param 'dataset' must be a subclass of Dataset and implement the abstract methods
		type: Dataset or its subclass

		param 'model' must be a subclass of Model and implement the abstract methods
		type: Model or its subclass

		param 'mode' takes the value of either 'Testing' or 'Validation'. 
		If 'Testing', the model is evaluated using the test set. 
		If 'Validation', the model is evaluated using validation set.

		param 'scheme' determines the proportion of the dataset (depending on 'mode', dataset can either be 1. test set or 2. validation set) to be used for as train tuples and test tuples during prequential evaluation. 
		E.g. [0.1, 0.9] will mean the chronological first 10% of each user's interaction will be labelled as train tuples, and the remaining 90% will be labelled as test tuples.
		'scheme' can also be the string 'LOO', which indicates Leave One Out and that only each user's last interaction will be labelled as test tuple, all other tuples are train tuples.
		type: list containing two floats/doubles/ints or a string

		Evaluate the given model using Prequential Evaluation and the given 'scheme', and then print the evaluation results. 
		Also return 'self.scores_list' (a list of dicts) which contains the evaluation results of the model.
		
		return type: a list of dicts

		"""

		if not isinstance(dataset, Dataset):
			raise ValueError('Your dataset object needs to be a subclass of Dataset and implements the abstract inner class.')

		if not (dataset.time_arranged_interactions_df['action'].apply(lambda x: isinstance(x, Dataset.Action) ).all() ):
			raise ValueError('The values inside the "action" column in your "dataset.time_arranged_interactions_df" is not of the "Dataset.Action" type. Please refer to "dataset.py". ')

		if not isinstance(model, Model):
			raise ValueError('Your model needs to be a subclass of Model and implements the abstract methods.')


		if type(scheme) == list:
			if (np.sum(scheme) != 1.0) and (len(scheme)== 2 ) :
				raise ValueError("The ratios indicated in 'scheme' does not add up to 1.")
		elif type(scheme) == str:
			if scheme != 'LOO':
				raise ValueError("'scheme' is an invalid string.")
		else: 
			raise ValueError("'scheme' is of an invalid type.")


		if mode == 'Testing':
			df = dataset.getTesting()
			df.reset_index(inplace=True)
			subset= 'test_set'
		elif mode == 'Validation':
			df = dataset.getValidation()
			df.reset_index(inplace=True)
			subset = 'validation_set'
		else:
			raise ValueError("mode is an invalid input. mode can only be the string 'Testing' or 'Validation'. ")

		df['index'] = df.index # store the index in a new column

		if type(scheme) == list:
			train_proportion = scheme[0]

			# split the tuples in df into Train tuples and Test tuples according to the 'scheme' specified.
			train_tuple_indices = df.groupby('userID').apply(lambda x: x.iloc[: round(len(x)*train_proportion) ])['index'].reset_index(drop=True)
			
			df['data_split_label'] = 'TestTuple' # temporarily label all tuples in the Test Set to be Test Tuples.
			df['data_split_label'].iloc[train_tuple_indices] = 'TrainTuple'

		elif type(scheme) == str:
			if scheme == 'LOO':
				# split the tuples in Testing Set into Train tuples and Test tuples according to the Leave-One-Out scheme
				test_tuple_indices = df.groupby('userID').apply(lambda x: x.iloc[-1])['index'].reset_index(drop=True)

				df['data_split_label'] = 'TrainTuple' # temporarily label all tuples in the Test Set to be Train Tuples.
				df['data_split_label'].iloc[test_tuple_indices] = 'TestTuple'
			else:
				raise NotImplementedError

		else:
			raise NotImplementedError



		# update the data split labels in the original dataset (rather than a copy of the dataset)
		dataset.updateTupleLabels( tupleLabels = df['data_split_label'] , subset = subset)

		"""

		'tuple' is a row of a Dataframe i.e. a Series
		'list_of_recommendations' is a Dataframe of length n, where each row represents a recommended item. The 1st row should be the top item that the model would like to recommend.

		"""

		for iteration_no in range( df.shape[0] ):
			tuple = dataset.getNextTuple()
			if tuple['data_split_label'] == 'TrainTuple':
				model.update_model(tuple)
			elif tuple['data_split_label'] == 'TestTuple':
				list_of_recommendations = model.test_model(tuple['userID'], tuple['time'])
				self.evaluateList(list_of_recommendations, tuple)

				model.update_model(tuple) # train on the test tuple after the evaluation has been done.
			else:
				raise NotImplementedError


		dataset.resetTupleLabels() # reset the data split labels in the original dataset.

		self.print_results()
		self.count_tuples_evaluated_on = 0 # reset the self.count_tuples_evaluated_on so that it will not affect the next evaluation.
		
		return self.scores_list # return the evaluation results of the model



