import os
import random
import enum
import numpy as np
import pandas as pd
from .dataset import Dataset

seed = 10
random.seed(seed)


class Amazon_Automotive_Dataset(Dataset):

	# implementing an abstract inner class from Parent class 'Action'
	class Action (Dataset.Action) :
		"""
		class Action is a Enum class that should contain all the possible actions between an user and an item.
		"""
		REVIEW = 1 # in Amazon Automotive Dataset, we assume each review indicates that the user has favourable preference towards the product. 


	def __init__(self, partition_ratios=[0.4, 0.2, 0.4] ,folder_path= 'amazon_automotive' ):

		super().__init__(partition_ratios = partition_ratios)

		users_interactions_filepath = os.path.join(folder_path, 'reviews_Automotive_5.json')

		self.interactions_df = pd.read_json(users_interactions_filepath, lines=True) 
		self.interactions_df = self.interactions_df.rename(columns={'reviewerID':'userID', 'asin':'itemID', 'unixReviewTime':'time', 'overall':'rating'} )
		self.interactions_df['action'] = self.Action.REVIEW # create a column of actions ('REVIEW' is the only possible action)
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

		df = df[ ['time', 'userID', 'itemID', 'action', 'rating'] ]
		df = df.sort_values('time', ascending=True)
		df.reset_index(drop=True, inplace=True)

		# split the df into train,validation, and test sets, and then return the resulting df.
		return self.split_dataset(df)



