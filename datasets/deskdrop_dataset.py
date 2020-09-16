import os
import random
import enum
import numpy as np
import pandas as pd
from .dataset import Dataset

seed = 10
random.seed(seed)


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


