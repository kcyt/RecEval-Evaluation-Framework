import os
import random
import numpy as np
import pandas as pd


class Dataset:

	def __init__(self, test_data_proportion = 0.2 ):

		"""
		param 'test_data_proportion': set the proportion of dataset to be used as test data. 
									 E.g. 0.2 means 20% of data is used as test set. 

		para, 'curr_pointer' shows the index of the next tuple to be retrieved 
		"""

		self.test_data_proportion = test_data_proportion
		self.curr_pointer = 0 


	def getNextTuple(self):

		"""
		This method must be implemented by a subclass.

		Return a tuple of the following format: <time, userID, itemID, action, train_test_label>
		"""

		pass




class Deskdrop_Dataset(Dataset):

	def __init__(self, test_data_proportion = 0.2, folder_path= 'CI&T_Deskdrop_dataset' ):

		super().__init__(test_data_proportion = test_data_proportion)

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

		# Generate the labels to indicate whether each tuple/row is a test or train tuple.
		num_of_obs = df.shape[0]
		test_size = round(self.test_data_proportion * num_of_obs)
		test_indices = random.sample( range(num_of_obs) ,test_size)
		test_bool_indices = np.isin(np.arange(num_of_obs), test_indices)
		train_test_labels = np.repeat('Train', repeats = num_of_obs)
		train_test_labels[test_bool_indices] = 'Test'
		df['train_test_label'] = train_test_labels

		return df



	def getNextTuple(self):
		"""
		Return a tuple <time, userID, itemID, action, train_test_label> that could either be a test or train tuple. 
		
		"""

		nextTuple = self.time_arranged_interactions_df.iloc[ self.curr_pointer , :]
		self.curr_pointer += 1

		return nextTuple









