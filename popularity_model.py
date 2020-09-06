import numpy as np
import pandas as pd 
from model import Model




class PopularityModel(Model):

	def __init__(self, n = 20, popularity_metric = 'IMDB_weighted_rating_formula' ):
		"""
		Attribute 'self.item_popularity_list' is a DataFrame used to keep track of the popularity of the items
		type: pd.DataFrame

		Attribute 'self.event_type_strength' will convert an interaction event into a 
				   corresponding score that indicates how much the user likes the item.
		type: int/float/double

		Attribute 'self.overall_avg_event_score' is the mean of all the average event score of all user-item interactions.
		type: float/double

		Attribute 'self.ninety_percentile_count' is the 90th percentile of the items' interaction counts.
		type: int/float/double

		Attribute 'self.popularity_metric' determines whether to use IMDB_weighted_rating_formula or simple interaction count to assess popularity.
		type: str

		Attribute 'self.compute_popularity_score' determines the function to be used to compute popularity score; it is determined by self.popularity_metric.
		type: function


		"""

		super().__init__(n = n)

		self.item_popularity_list = None

		# the self.event_type_strength dict assumes the dataset used is Deskdrop dataset.
		self.event_type_strength = {
				'VIEW': 1.0,
				'LIKE': 2.0, 
				'BOOKMARK': 2.5, 
				'FOLLOW': 3.0,
				'COMMENT': 4.0,  
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

		if self.item_popularity_list is None:  # Model is given its first tuple to be used for training.
			self.item_popularity_list = pd.DataFrame([[ str(reduced_train_tuple['time']) , reduced_train_tuple['itemID'],  reduced_train_tuple['action'].name  ]], columns = ['time', 'itemID', 'action'] )
			
			self.item_popularity_list['interaction_count'] = 1
			item_average_event_score = self.event_type_strength[ reduced_train_tuple['action'].name ]
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
			updated_average_event_score = ( current_item_tuple['interaction_count'] * current_item_tuple['average_event_score'] + self.event_type_strength[ reduced_train_tuple['action'].name ] ) / (current_item_tuple['interaction_count'] + 1 )
			updated_interaction_count = current_item_tuple['interaction_count'] + 1

			# assign the updated_average_event_score
			self.item_popularity_list.loc[ item_index, 'average_event_score' ] = updated_average_event_score
			
			# assign the updated_interaction_count
			self.item_popularity_list.loc[ item_index, 'interaction_count' ] = updated_interaction_count

			# append the list (stored as a string) of timestamps for interactions with that item
			self.item_popularity_list.loc[ item_index, 'time' ] = current_item_tuple['time'] + ' ' + str(reduced_train_tuple['time'])

			# append to the list (stored as a string) of actions for that item
			self.item_popularity_list.loc[ item_index, 'action' ] = current_item_tuple['action'] + ' ' + reduced_train_tuple['action'].name


	def IMDB_weighted_rating_formula(self, row ): 

		"""
		param 'row': a row of the self.item_popularity_list
		type: pd.Series

		Compute a popularity score for an item given the current average event score and interaction count.
		The popularity score formula is derived from the weighted rating formula used in IMDB's Top 250 to evaluate a movie's popularity.
		
		return type: float/double

		"""
		item_interaction_count = row['interaction_count']
		item_average_event_score = row['average_event_score']

		return (item_interaction_count/(item_interaction_count+self.ninety_percentile_count)*item_average_event_score + self.ninety_percentile_count/(item_interaction_count+self.ninety_percentile_count)*self.overall_avg_event_score )


	def simple_interaction_count(self, row ): 

		"""
		param 'row': a row of the self.item_popularity_list
		type: pd.Series 

		Compute a popularity score for an item using just interaction count of that item.
		The popularity score formula is simply taking the interaction count.
		
		return type: int

		"""

		return row['interaction_count']


	def test_model(self, test_tuple_userID, test_tuple_timestamp):

		"""
		param 'test_tuple_userID': userID of the test tuple - userID is not relevant/useful in Popularity Model
		type: int

		param 'test_tuple_timestamp' is unix timestamp of the user-item interaction represented by the test tuple. It is not used by the Popularity Model as well.
		type: int 


		1. Compute the popularity score of each item (Need to be re-computed whenever the model is given a new test instance 
		since every train instance can change the popularity score of every item).
		2. Sort the item_popularity_list according to their popularity score before getting the top n recommended items

		Return a dataframe containing the top n items that is recommended by the popularity model.

		Return type: pd.Dataframe

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

		return self.item_popularity_list.head(self.n).copy()



	def train_in_batch(self, dataset):
		"""
		param 'dataset' is the data samples used for training of the model. (e.g. training set, or concatentation of training set and validation set)
		type: pd.DataFrame

		To train the model using 'dataset'. This function will compute self.item_popularity_list, which tracks the popular items in the dataset.
		
		No return object.

		"""

		time_column = dataset.groupby('itemID')['time'].apply(lambda x: ' '.join(str(i) for i in x) )
		itemID_column = pd.Series(time_column.index.tolist() )
		time_column.reset_index(drop=True, inplace=True)
		interaction_count_column = dataset.groupby('itemID')['time'].count().reset_index(drop=True)
		action_column = dataset.groupby('itemID')['action'].apply(lambda x: ' '.join(i.name for i in x) ).reset_index(drop=True)
		average_event_score_column = dataset.groupby('itemID')['action'].apply(lambda x: np.sum( [self.event_type_strength[i.name] for i in x] )/len(x)  ).reset_index(drop=True)
		self.item_popularity_list = pd.concat( [time_column, itemID_column, action_column, interaction_count_column, average_event_score_column  ], axis = 1, ignore_index= True ).rename(columns = {0:'time', 1:'itemID', 2:'action', 3:'interaction_count', 4:'average_event_score'} )


		# compute the self.overall_avg_event_score and self.ninety_percentile_count as their values are needed to compute popularity score if 'IMDB_weighted_rating_formula' is used. 
		self.overall_avg_event_score = self.item_popularity_list['average_event_score'].mean() 
		self.ninety_percentile_count = self.item_popularity_list['interaction_count'].quantile(0.90)

		# Compute popularity score of each item and return top n most popular items.
		self.item_popularity_list['popularity_score'] = self.item_popularity_list.apply(func = self.compute_popularity_score, axis=1 )
		self.item_popularity_list = self.item_popularity_list.sort_values('popularity_score', ascending=False)



	def predict_in_batch(self, validation_set):
		"""
		param 'validation_set' is the data samples used for validation of the model.
		type: pd.DataFrame

		For each user in 'validation_set', predict the top n item recommended by the model.
		Return a dataframe containing the recommendations for each user. The dataframe adheres to the following format:
			The dataframe will have 3 columns: 1. 'userID' 2. 'itemID_list', and 3.'action_list'
			The 'itemID_list' column contains a list of recommended itemID for each user.
			The 'action_list' column contains a list of recommended action for each user. Each action must be an enum defined in a subclass of dataset.Action 
		
		return type: pd.DataFrame


		"""

		userID_list = validation_set['userID'].unique()
		recommendation_df = pd.DataFrame(userID_list, columns=['userID'])

		top_n_recommended_items = self.item_popularity_list.head(self.n).copy()
		top_n_recommended_items_string = " ".join( [str(itemID) for itemID in top_n_recommended_items['itemID'] ] ) # Note that the top n recommended items are stored as a string instead of a list.
		recommendation_df['itemID_list'] = top_n_recommended_items_string # each user will be recommended the same n items.
		recommendation_df['itemID_list'] = recommendation_df['itemID_list'].apply(lambda x: list(map( int , x.split(' ') )) )
		recommendation_df['action_list'] = None	 # no value is needed to be given to action_list if prediction of action is not required.

		return recommendation_df





