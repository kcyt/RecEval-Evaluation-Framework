import numpy as np
import pandas as pd 
from .model import Model
from surprise import SVD as surprise_SVD
from surprise import Dataset as surprise_dataset
from surprise import Reader as surprise_reader
from collections import defaultdict

"""
Implementation of the SVD model using the surprise library.
"""


class SVD_Model(Model):

	def __init__(self, n = 20, time_delay_from_t0 = 180,  num_of_interval = 6  ):
		"""
		Attribute 'self.accumulative_tuple_df' is a DataFrame used to contain the tuples which have been given to SVD_Model for training. 
		type: pd.DataFrame

		Attribute 'self.event_type_strength' will convert an interaction event into a 
				   corresponding score that indicates how much the user likes the item.
		type: int/float/double


		Attribute 'self.engine' is an instance of the SVD class from surprise library. It functions as the underlying engine for the SVD_Model.
		type: The SVD class from surprise library.


		"""

		super().__init__(n = n)

		self.accumulative_tuple_df = None

		# the self.event_type_strength dict assumes the dataset used is Deskdrop dataset.
		self.event_type_strength = {
				'VIEW': 1.0,
				'LIKE': 2.0, 
				'BOOKMARK': 2.5, 
				'FOLLOW': 3.0,
				'COMMENT': 4.0,  
				} 

		self.engine = surprise_SVD()







	def update_model(self, train_tuple):

		if self.accumulative_tuple_df is None: # model is given its first tuple for training
			self.accumulative_tuple_df = pd.DataFrame(train_tuple).T
			self.accumulative_tuple_df.reset_index(inplace=True,drop=True)
		else:
			len_of_accumulative_tuple_df = self.accumulative_tuple_df.shape[0]
			self.accumulative_tuple_df.loc[len_of_accumulative_tuple_df] = train_tuple





	def test_model(self, test_tuple_userID, test_tuple_timestamp):

		"""
		param 'test_tuple_userID': userID of the test tuple. 
		type: int

		param 'test_tuple_timestamp' is unix timestamp of the user-item interaction represented by the test tuple. 
		type: int 

		Return a dataframe containing the top n items that is recommended by the SVD_Model model.

		Return type: pd.Dataframe

		"""

		# if first tuple given to the model is a test tuple, then the SVD_Model model will not be able to make any recommendations
		if self.accumulative_tuple_df is None:
			return None 

		# construct a train set in the format required by the surprise package.
		df = self.accumulative_tuple_df[['itemID', 'userID']]
		df['rating'] =  self.accumulative_tuple_df['action'].apply(func=lambda x: self.event_type_strength[x.name])
		reader = surprise_reader(rating_scale=(1.0, 4.0))
		data = surprise_dataset.load_from_df(df, reader)
		trainset = data.build_full_trainset()

		# train the SVD_Model model.
		self.engine.fit(trainset) 

		# generate predictions for test_tuple_userID by assessing how interested 
		# the user will be in all the items that the model has seen thus far.
		test_df = pd.DataFrame( {'itemID': df['itemID'].unique() } )
		test_df['userID'] = test_tuple_userID
		test_df['rating'] = 1.0 # just a placeholder
		testset = surprise_dataset.load_from_df(test_df, reader)
		testset = testset.build_full_trainset().build_testset()
		predictions = self.engine.test(testset)

		# Sorting the predictions and getting the top n for the given user
		top_n = defaultdict(list)
		for iid, uid, true_r, est, _ in predictions:
			top_n[uid].append((iid, est))

		for uid, user_ratings in top_n.items():
			user_ratings.sort(key=lambda x: x[1], reverse=True)
			top_n[uid] = user_ratings[:self.n]

		top_n_recommendations = [x[0] for x in top_n[test_tuple_userID] ]
		top_n_recommendations = pd.DataFrame( {'itemID': pd.Series(top_n_recommendations)} )

		return top_n_recommendations

			
		

	def train_in_batch(self, dataset):
		if self.accumulative_tuple_df is not None:
			print("Note: 'train_in_batch()' is called, and the data samples given to decay_popularity_model for training have been replaced")
		
		self.accumulative_tuple_df = dataset.copy()








