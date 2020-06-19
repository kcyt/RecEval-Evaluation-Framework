import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

seed = 10
random.seed(seed)

class Dataset:

	# the __init__ function must be called by the Subclass's own '__init__' function.
	def __init__(self, test_data_proportion = 0.2, kfold_num_splits = 5 ):

		"""
		Attribute 'self.test_data_proportion': set the proportion of dataset to be used as test data. 
									 E.g. 0.2 means 20% of data is used as test set. 
		type: float/double in the interval (0,1)

		Attribute 'self.curr_pointer' shows the index of the next tuple to be retrieved 
		type: int

		Attribute 'self.kfold_num_splits': number of k-folds to use. If set to 0 or 1, then kfold will not be used.
		type: int

		Attribute 'self.time_arranged_interactions_df' is a dataframe containing user-item interactions where each row is a tuple of the following format: <time, userID, itemID, action, k_fold_1, ... , k_fold_last , data_split_label>

		"""

		self.test_data_proportion = test_data_proportion
		self.curr_pointer = 0 
		self.kfold_num_splits = kfold_num_splits

		# the following attributes must be set by the Subclass.
		self.time_arranged_interactions_df = None # must be set in order to use the self.switch_k_fold() function


	# Abstract method
	def getNextTuple(self):

		"""
		This method must be implemented by a subclass.

		Return a tuple of the following format: <time, userID, itemID, action, k_fold_1, ... , k_fold_last , data_split_label>
		Return type: pd.Series object
		"""

		raise NotImplementedError("Please override and implement the 'getNextTuple(self)' method in your dataset object")


	# helper method to split a time ordered df into training set, validation set, and test set
	def split_dataset(self, time_ordered_df):

		"""
		Split a dataset into training set, validation set, and test set by giving data split labels to each data sample
		Note that when a data split label will be either 'Train', 'Valid' or 'Test', and it means the data sample is in training set, validation set, or test set respectively.

		param 'time_ordered_df': a df containing all the data samples that are ordered by time (i.e. earliest sample take index 0)
		type: pd.DataFrame object

		Return a time ordered df that has an additional column 'data_split_label', which specifies whether the sample is in train, validation, or test set.
		If K-fold is used, then create additional columns for each fold as well.
		Return type: pd.DataFrame object
		"""

		# Generate the labels to indicate whether each tuple/row in df is a train, validation, or test tuple.
		num_of_obs = time_ordered_df.shape[0]
		test_size = round(self.test_data_proportion * num_of_obs)
		test_indices = random.sample( range(num_of_obs) ,test_size)
		test_bool_indices = np.isin(np.arange(num_of_obs), test_indices)
		non_test_indices = np.array( list( set(range(num_of_obs)) - set(test_indices) ) )

		# KFold is used to split non-test samples into train and validation sets
		num_split = self.kfold_num_splits if self.kfold_num_splits > 1 else 2 # If user wants to do train-validation split without using K-fold, then 2-Fold split is still carried out and the 1st fold is used as the train-validation set.
		kf = KFold(n_splits = num_split , shuffle = True, random_state = seed )

		data_split_labels = np.repeat('Train', repeats = num_of_obs) # temporarily set all samples to have the 'Train' label
		data_split_labels[test_bool_indices] = 'Test' 

		# for each K-fold, create a column in the time_ordered_df containing labels indicating whether the row is in train,validation, or test set.
		for k_fold_index , (train_index, validation_index) in enumerate(kf.split(non_test_indices) , 1 ) :
			current_data_split_labels = data_split_labels.copy()

			actual_validation_index = non_test_indices[validation_index] # actual_validation_index is the actual indices of the time_ordered_df
			actual_validation_bool_indices = np.isin(np.arange(num_of_obs), actual_validation_index)
			current_data_split_labels[actual_validation_bool_indices] = 'Valid'

			k_fold_column_name = 'k_fold_' + str(k_fold_index)
			time_ordered_df[k_fold_column_name] = current_data_split_labels

		# set the current train-validation sets to be the 1st K-fold
		time_ordered_df['data_split_label'] = time_ordered_df['k_fold_1'] 

		return time_ordered_df



	# helper method to switch the training-validation sets into another K-fold.
	def switch_k_fold(self, k_fold_index):
		"""
		Required: The attribute 'self.time_arranged_interactions_df' must be already set.

		Use another K-fold by switching the 'data_split_label' column inside 'self.time_arranged_interactions_df' with another 'k_fold' column

		param 'k_fold_index': indicates which of the k folds to use.
		type: int


		"""
		k_fold_column_name = 'k_fold_' + str(k_fold_index)
		self.time_arranged_interactions_df['data_split_label'] = self.time_arranged_interactions_df[k_fold_column_name] 







class Deskdrop_Dataset(Dataset):

	def __init__(self, test_data_proportion = 0.2, kfold_num_splits = 5, folder_path= 'CI&T_Deskdrop_dataset' ):

		super().__init__(test_data_proportion = test_data_proportion, kfold_num_splits = kfold_num_splits)

		shared_articles_filepath =  os.path.join(folder_path, 'shared_articles.csv')
		users_interactions_filepath = os.path.join(folder_path, 'users_interactions.csv')

		self.articles_df = pd.read_csv(shared_articles_filepath,low_memory=False)
		self.interactions_df = pd.read_csv(users_interactions_filepath,low_memory=False)
		self.time_arranged_interactions_df = self.process_time_arranged_df(self.interactions_df)


	def process_time_arranged_df(self, df ):
		"""
		Given an user-item interaction df, remove unnecessary columns, do renaming, 
		sort according to timestamp, and generate train/test labels for the tuples. 

		param 'df': the user-item interaction dataframe
		Return a time-arranged df
		
		"""


		df = df[ ['timestamp', 'personId', 'contentId', 'eventType'] ]
		df = df.rename(columns={'timestamp':'time', 'personId':'userID', 'contentId':'itemID', 'eventType': 'action' } )
		df = df.sort_values('time', ascending=True)


		# split the df into train,validation, and test sets, and then return the resulting df.
		return self.split_dataset(df)




	def getNextTuple(self):
		"""
		Return a tuple <time, userID, itemID, action, k_fold_1, ... , k_fold_last , data_split_label> that could either be a test or train tuple. 
		
		"""

		nextTuple = self.time_arranged_interactions_df.iloc[ self.curr_pointer , :]
		self.curr_pointer += 1

		return nextTuple









