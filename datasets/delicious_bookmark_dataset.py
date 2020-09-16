import os
import random
import enum
import numpy as np
import pandas as pd
from .dataset import Dataset

seed = 10
random.seed(seed)


class Delicious_Bookmark_Dataset(Dataset):

	# implementing an abstract inner class from Parent class 'Action'
	class Action (Dataset.Action) :
		"""
		class Action is a Enum class that should contain all the possible actions between an user and an item.
		"""
		TAG = 1 # in Delicious, we have to assume every type of tag is equivalent, and every tag indicates a user's preference for a URL. 


	def __init__(self, partition_ratios=[0.4, 0.2, 0.4] ,folder_path= 'delicious_social_bookmarking' ):

		super().__init__(partition_ratios = partition_ratios)

		users_interactions_filepath = os.path.join(folder_path, 'user_taggedbookmarks-timestamps.dat')

		self.interactions_df = pd.read_csv(users_interactions_filepath, sep='\t', low_memory=False) # itemID corresponds to bookmarkID in the Delicious Bookmark dataset.
		self.interactions_df = self.interactions_df.rename(columns={'bookmarkID':'itemID', 'timestamp':'time'} )
		self.interactions_df['action'] = self.Action.TAG # create a column of actions ('TAG' is the only possible action)
		self.time_arranged_interactions_df = self.process_time_arranged_df(self.interactions_df)

	def process_time_arranged_df(self, df ):
		"""
		param 'df': the user-item interaction dataframe
		type: pd.DataFrame

		Given an user-item interaction df, reorder the columns, 
		sort according to timestamp, and generate train/validation/test labels for the tuples. 
		Return a time-arranged df.

		Return type: pd.DataFrame
		
		"""

		df = df[ ['time', 'userID', 'itemID', 'action', 'tagID'] ]
		df = df.sort_values('time', ascending=True)
		df.reset_index(drop=True, inplace=True)

		# split the df into train,validation, and test sets, and then return the resulting df.
		return self.split_dataset(df)

