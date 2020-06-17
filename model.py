import numpy as np
import pandas as pd 


class Model:

	def __init__(self, n = 20):

		"""
		param 'n' refers to the top n recommendations to be returned by the Model
		"""

		self.n = n 
 

	def update_model(self, train_tuple):
		"""
		param 'train_tuple': is the tuple given to the model for training purpose.
							 format of a tuple: <time, userID, itemID, action, train_test_label>

		Train/Fine-tune the model given 1 train tuple


		"""

		pass


	def test_model(self, test_tuple_userID):
		"""

		Retrieve a ranked list of items based only on the userID of the test_tuple

		"""
		pass


class PopularityModel(Model):

	def __init__(self, n = 20, popularity_metric = 'IMDB_weighted_rating_formula' ):
		"""
		Attribute 'self.item_popularity_list' is a DataFrame used to keep track of the popularity of the items
		Attribute 'self.event_type_strength' will convert an interaction event into a 
				   corresponding score that indicates how much the user likes the item.
		Attribute 'self.overall_avg_event_score' is the mean of all the average event score of all user-item interactions.
		Attribute 'self.ninety_percentile_count' is the 90th percentile of the items' interaction counts.
		Attribute 'self.popularity_metric' determines whether to use IMDB_weighted_rating_formula or simple interaction count to assess popularity.
		Attribute 'self.compute_popularity_score' determines the function to be used to compute popularity score; it is determined by self.popularity_metric.
		"""

		super().__init__(n = n)

		self.item_popularity_list = None
		self.event_type_strength = {
				'VIEW': 1.0,
				'LIKE': 2.0, 
				'BOOKMARK': 2.5, 
				'FOLLOW': 3.0,
				'COMMENT CREATED': 4.0,  
				} 
		self.overall_avg_event_score = 0
		self.ninety_percentile_count = 0
		self.popularity_metric = popularity_metric

		if self.popularity_metric == "IMDB_weighted_rating_formula" :
			self.compute_popularity_score = self.IMDB_weighted_rating_formula
		elif self.popularity_metric == "simple interaction count" :
			self.compute_popularity_score = self.simple_interaction_count
		else:
			raise NotImplementedError



	def update_model(self, train_tuple):

		reduced_train_tuple = train_tuple[ ['time', 'itemID', 'action' ] ] # remove columns not needed for training the model

		if self.item_popularity_list is None:
			self.item_popularity_list = pd.DataFrame([[ str(reduced_train_tuple['time']) , reduced_train_tuple['itemID'],  str(reduced_train_tuple['action'])  ]], columns = ['time', 'itemID', 'action'] )
			
			self.item_popularity_list['interaction_count'] = 1
			item_average_event_score = self.event_type_strength[ reduced_train_tuple['action'] ]
			self.item_popularity_list['average_event_score'] = item_average_event_score
			self.item_popularity_list['popularity_score'] = 0 # temporarily set as 0, the actual score will be computed during model testing.
		else:

			# Check if item exists in the item_popularity_list
			item_index = self.item_popularity_list['itemID'] == reduced_train_tuple['itemID']
			if ( (~item_index).all() ): # if itemID not in the self.item_popularity_list, create a new row with placeholder data elements to be filled later.
				self.item_popularity_list.loc[ len(self.item_popularity_list) ] = ['', reduced_train_tuple['itemID']  , '' , 0, 0, 0 ] 
				item_index = self.item_popularity_list['itemID'] == reduced_train_tuple['itemID'] # update the item_index

			# get the tuple corresponding to the item in the given train_tuple
			current_item_tuple = self.item_popularity_list[ item_index ].iloc[0]
			
			# compute the updated_average_event_score and updated_interaction_count
			updated_average_event_score = ( current_item_tuple['interaction_count'] * current_item_tuple['average_event_score'] + self.event_type_strength[ reduced_train_tuple['action'] ] ) / (current_item_tuple['interaction_count'] + 1 )
			updated_interaction_count = current_item_tuple['interaction_count'] + 1

			# assign the updated_average_event_score
			self.item_popularity_list.loc[ item_index, 'average_event_score' ] = updated_average_event_score
			
			# assign the updated_interaction_count
			self.item_popularity_list.loc[ item_index, 'interaction_count' ] = updated_interaction_count

			# append the list of time interacted for that item
			self.item_popularity_list.loc[ item_index, 'time' ] = current_item_tuple['time'] + ' ' + str(reduced_train_tuple['time'])

			# append to the list of actions for that item
			self.item_popularity_list.loc[ item_index, 'action' ] = current_item_tuple['action'] + ' ' + str(reduced_train_tuple['action'])


	def IMDB_weighted_rating_formula(self, row ): 

		"""

		Compute a popularity score for an item given the current average event score and interaction count.
		The popularity score formula is derived from the weighted rating formula used in IMDB's Top 250 to evaluate a movie's popularity.
		
		param 'row': a row of the self.item_popularity_list

		"""
		item_interaction_count = row['interaction_count']
		item_average_event_score = row['average_event_score']

		return (item_interaction_count/(item_interaction_count+self.ninety_percentile_count)*item_average_event_score + self.ninety_percentile_count/(item_interaction_count+self.ninety_percentile_count)*self.overall_avg_event_score )


	def simple_interaction_count(self, row ): 

		"""

		Compute a popularity score for an item given the current average event score and interaction count.
		The popularity score formula is simply taking the interaction count.
		
		param 'row': a row of the self.item_popularity_list

		"""

		return row['interaction_count']


	def test_model(self, test_tuple_userID):

		"""
		1. Compute the popularity score of each item (Need to be re-computed whenever the model is given a new test instance 
		since every train instance can change the popularity score of every item).
		2. Sort the item_popularity_list according to their popularity score before getting the top n recommended items

		param 'test_tuple_userID': userID of the test tuple - userID is not relevant/useful in Popularity Model

		Return a dataframe containing the top n items that is recommended by the popularity model.

		"""

		# if first tuple given to the model is a test tuple, then the popularity model will not be able to make any recommendations
		if self.item_popularity_list is None:
			return None 


		# re-compute the self.overall_avg_event_score and self.ninety_percentile_count as their values will be changed after model training.
		self.overall_avg_event_score = self.item_popularity_list['average_event_score'].mean() 
		self.ninety_percentile_count = self.item_popularity_list['interaction_count'].quantile(0.90)

		# Compute popularity score of each item and return top n most popular items.
		self.item_popularity_list['popularity_score'] = self.item_popularity_list.apply(func = self.compute_popularity_score, axis=1 )
		self.item_popularity_list = self.item_popularity_list.sort_values('popularity_score', ascending=False)

		return self.item_popularity_list.head(self.n)


