import os
import random
import enum
import numpy as np
import pandas as pd

seed = 10
random.seed(seed)

class Dataset:

	# an abstract inner class that must be inherited by an inner class inside a Subclass of Dataset. 
	class Action (enum.Enum) :
		"""
		class Action is a Enum class that should contain all the possible actions between an user and an item.
		
		e.g. of a sample code:
		LIKE = 1
		BOOKMARK = 2
		COMMENT = 3
		SHARE = 4

		"""
		pass
		


	# the __init__ function must be called by the Subclass's own '__init__' function.
	def __init__(self, partition_ratios = [0.4, 0.2, 0.4]):

		"""
		Attribute 'self.partition_ratios' is a list containing 3 float/double elements indicating the 
		proportion of the dataset to be in training, validation, and testing sets represectively.
		type: list of floats/doubles

		Attribute 'self.train_size' is the size of the training set
		Note: This attribute is only set after calling split_dataset(), and this attribute must be set before calling getTraining() and getValidation().
		type: int 

		Attribute 'self.train_and_validation_size' is the size of the training set and validation set combined.
		Note: This attribute is only set after calling split_dataset(), and this attribute must be set before calling getValidation() and calling getTesting().
		type: int 

		Attribute 'self.curr_pointer' shows the index of the next tuple to be retrieved 
		type: int

		Attribute 'self.time_arranged_interactions_df' is a dataframe containing user-item interactions where each row is a tuple of the following format: <time, userID, itemID, action, data_split_label>
		type: pd.DataFrame

		Attribute 'self.initial_data_split_label' is a np.array that contains the initial data split labels for each row in self.time_arranged_interactions_df.

		"""
		if np.sum(partition_ratios) != 1.0 :
			raise ValueError("partition_ratios does not add up to 1!")

		self.partition_ratios = partition_ratios
		self.train_size = None 
		self.train_and_validation_size = None
		self.curr_pointer = None
		self.time_arranged_interactions_df = None 
		self.initial_data_split_label = None



	def getNextTuple(self):
		"""
		Return a tuple of the following format <time, userID, itemID, action, data_split_label> that could either be a train or test tuple. 
		Note that this function requires the use of self.curr_pointer, which is initialized by the self.set_curr_pointer() function. Rather than calling split_dataset(), it is equally appropriate
		to manually set the self.curr_pointer, which should be initialized to be the index of the first validation sample in self.time_arranged_interactions_df if doing validation, 
		and should be initialized to index of first test sample if doing testing.

		Return type: pd.Series object

		"""

		nextTuple = self.time_arranged_interactions_df.iloc[ self.curr_pointer , :]
		self.curr_pointer += 1

		return nextTuple


	# helper method to split a time ordered df into training set, validation set, and test set
	def split_dataset(self, time_ordered_df ):

		"""
		param 'time_ordered_df': a df containing all the data samples that are ordered by time (i.e. earliest sample take index 0)
		type: pd.DataFrame object


		Split a dataset into training set, validation set, and test set by giving data split labels to each data sample
		Note that when a data split label will be either 'Train', 'Valid' or 'Test', and it means the data sample is in training set, validation set, or test set respectively.
		Return a time ordered df that has an additional column 'data_split_label', which specifies whether the sample is in train, validation, or test set.
		This function will also set self.train_size and self.train_and_validation_size, which are both int type and indicate the size of training set and the size of training + validation set respectively
		This function will also set self.initial_data_split_label, which is needed before 'self.updateTupleLabels()' can be called.

		Return type: pd.DataFrame object
		"""


		train_data_proportion = self.partition_ratios[0]
		train_and_validation_data_proportion = train_data_proportion + self.partition_ratios[1]


		# Generate the labels to indicate whether each tuple/row in df is a train, validation, or test tuple.
		num_of_obs = time_ordered_df.shape[0]
		self.train_size = round(train_data_proportion * num_of_obs)
		self.train_and_validation_size = round(train_and_validation_data_proportion * num_of_obs)

		data_split_boolean_labels = np.repeat(False, repeats = num_of_obs)
		train_boolean_labels = data_split_boolean_labels.copy()
		train_boolean_labels[:self.train_size] = True
		validation_boolean_labels = data_split_boolean_labels.copy()
		validation_boolean_labels[self.train_size:self.train_and_validation_size] = True
		test_boolean_labels = data_split_boolean_labels.copy()
		test_boolean_labels[self.train_and_validation_size:] = True
		
		data_split_labels = np.repeat('Train', repeats = num_of_obs) # temporarily set all samples to have the 'Train' label
		data_split_labels[validation_boolean_labels] = 'Valid'
		data_split_labels[test_boolean_labels] = 'Test'
	
		time_ordered_df['data_split_label'] = data_split_labels
		self.initial_data_split_label = data_split_labels.copy() # To be used for reseting the data_split_labels.

		return time_ordered_df

	def set_curr_pointer(self, mode='Testing'):

		"""
		Note: To use this method, the self.train_and_validation_size and self.train_size must be set. They are set by calling the self.split_dataset() function.

		param 'mode' can either be 'Testing' or 'Validation'.
		If 'Testing', the self.curr_pointer is set to the index of the first data sample in the Testing set.
		If 'Validation', the self.curr_pointer is set to the index of the first data sample in the Validation set.

		This method will set the attribute self.curr_pointer

		"""
		if mode == 'Testing':
			self.curr_pointer = self.train_and_validation_size
		elif mode == 'Validation':
			self.curr_pointer = self.train_size
		else:
			raise ValueError("The 'mode' argument given to 'set_curr_pointer()' is invalid. It can only be a string of 'Testing' or 'Validation'.")




	def getTraining(self):

		"""

		Return the Training Set from the overall Dataset.
		Note: This function requires the use of self.train_size, which is set only after calling the split_dataset() function. 
		To use getTraining(), instead of calling split_dataset(), it is just as appropriate to simply set self.train_size manually.

		return type: pd.DataFrame

		"""
		return self.time_arranged_interactions_df.iloc[:self.train_size ,:].copy()

		

	def getValidation(self, exclude_first_time_users = False):
		"""

		param 'exclude_first_time_users': if set to True, would return a Validation Set that contains no first_time_users i.e. the 
		returned Validation Set would contain only users that have appeared in the Training Set.
		type: boolean

		Return the Validation Set from the overall Dataset.
		Note: This function requires the use of self.train_size and self.train_and_validation_size, which are set only after calling the split_dataset() function. 
		To use getValidation(), instead of calling split_dataset(), it is just as appropriate to simply set self.train_size and self.train_and_validation_size manually.

		return type: pd.DataFrame
		"""

		validation_set = self.time_arranged_interactions_df.iloc[self.train_size:self.train_and_validation_size ,:]
		
		if not exclude_first_time_users: # do not exclude first time users
			return validation_set

		else: # exclude first time users
			training_set = self.getTraining()
			unique_userID = training_set['userID'].unique()

			return validation_set.loc[validation_set['userID'].isin(unique_userID) ].copy()

	def getTesting(self):
		"""

		Return the Testing Set from the overall Dataset.
		Note: This function requires the use of self.train_and_validation_size, which is set only after calling the split_dataset() function. 
		To use getTesting(), instead of calling split_dataset(), it is just as appropriate to simply set self.train_and_validation_size manually.

		return type: pd.DataFrame
		"""
		return self.time_arranged_interactions_df.iloc[self.train_and_validation_size: ,:].copy()

	def updateTupleLabels(self, tupleLabels, subset= 'test_set'):

		"""
		param 'tupleLabels': a column of new tuple labels for either the 1.Testing set or 2.Validation set.
		type: pd.Series

		param 'subset' can either take the value of 'test_set' or 'validation_set'.
		If 'test_set', only the Testing set will be updated.
		If 'validation_set', only Validation set will be updated.

		Update the Tuple labels in either 1.Testing set or 2.Validation set.

		No return object.

		"""

		if subset == 'test_set':
			if len(tupleLabels) !=  (len(self.time_arranged_interactions_df) - self.train_and_validation_size):
				raise ValueError("length of 'tupleLabels' is inconsistent with the Testing Set ")
			
			updated_tuple_labels = self.time_arranged_interactions_df['data_split_label'].iloc[:self.train_and_validation_size]
			updated_tuple_labels = pd.concat([updated_tuple_labels, tupleLabels], axis=0, ignore_index=True)
			self.time_arranged_interactions_df['data_split_label'] =  updated_tuple_labels

		elif subset == 'validation_set':
			validation_size = self.train_and_validation_size - self.train_size
			if len(tupleLabels) !=  validation_size :
				raise ValueError("length of 'tupleLabels' is inconsistent with the Validation set ")
			
			updated_tuple_labels_1 = self.time_arranged_interactions_df['data_split_label'].iloc[:self.train_size]
			updated_tuple_labels_2 = self.time_arranged_interactions_df['data_split_label'].iloc[self.train_and_validation_size:]
			updated_tuple_labels = pd.concat([updated_tuple_labels_1, tupleLabels, updated_tuple_labels_2], axis=0, ignore_index=True)
			self.time_arranged_interactions_df['data_split_label'] =  updated_tuple_labels

		else:
			raise ValueError("'subset' must be a string containing either 'test_set' or 'validation_set'")



	def resetTupleLabels(self):
		"""
		Note: 'resetTupleLabels()' requires 'self.initial_data_split_label' to be set, and it is set only when 'split_dataset()' is called.
		It is equally appropriate for self.initial_data_split_label to be set manually somewhere in custom codes written by users of this framework.
		The attribute 'self.initial_data_split_label' is a np.array that contains the initial data split labels for each row in self.time_arranged_interactions_df.

		To reset the Tuple labels in the entire dataset to their initial values.

		No return object

		"""
		if self.initial_data_split_label is None:
			raise Exception("self.initial_data_split_label is not set. It is onlt set when split_dataset() is called.")

		self.time_arranged_interactions_df['data_split_label'] =  self.initial_data_split_label.copy()







class Deskdrop_Dataset(Dataset):

	# implementing an abstract inner class from Parent class 'Action'
	class Action (Dataset.Action) :
		"""
		class Action is a Enum class that should contain all the possible actions between an user and an item.
		"""
		VIEW = 1
		LIKE = 2 
		BOOKMARK = 3 
		FOLLOW = 4
		COMMENT = 5  



	def __init__(self, partition_ratios=[0.4, 0.2, 0.4] ,folder_path= 'CI&T_Deskdrop_dataset' ):

		super().__init__(partition_ratios = partition_ratios)

		shared_articles_filepath =  os.path.join(folder_path, 'shared_articles.csv')
		users_interactions_filepath = os.path.join(folder_path, 'users_interactions.csv')

		self.articles_df = pd.read_csv(shared_articles_filepath,low_memory=False)
		self.interactions_df = pd.read_csv(users_interactions_filepath,low_memory=False)
		self.time_arranged_interactions_df = self.process_time_arranged_df(self.interactions_df)

		# convert the 'action' column in 'self.time_arranged_interactions_df' from str type into our defined enum type:
		self.Action.str_to_enum_dict = {'VIEW': self.Action.VIEW, 'LIKE': self.Action.LIKE, 
										'BOOKMARK': self.Action.BOOKMARK, 'FOLLOW': self.Action.FOLLOW, 
										'COMMENT CREATED': self.Action.COMMENT,  }
		self.time_arranged_interactions_df['action'] = self.time_arranged_interactions_df['action'].apply(lambda x: self.Action.str_to_enum_dict[x] ) 


	def process_time_arranged_df(self, df ):
		"""
		param 'df': the user-item interaction dataframe
		type: pd.DataFrame

		Given an user-item interaction df, remove unnecessary columns, do renaming, 
		sort according to timestamp, and generate train/validation/test labels for the tuples. 
		Return a time-arranged df.

		Return type: pd.DataFrame
		
		"""


		df = df[ ['timestamp', 'personId', 'contentId', 'eventType'] ]
		df = df.rename(columns={'timestamp':'time', 'personId':'userID', 'contentId':'itemID', 'eventType': 'action' } )
		df = df.sort_values('time', ascending=True)
		df.reset_index(drop=True, inplace=True)


		# split the df into train,validation, and test sets, and then return the resulting df.
		return self.split_dataset(df)












