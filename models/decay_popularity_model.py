import numpy as np
import pandas as pd 
from .model import Model

"""
Implementation of the DecayPopularityModel is based on the algorithm provided by the paper 'A Re-visit of the Popularity Baseline in Recommender Systems' by Yitong Ji
"""


class DecayPopularityModel(Model):

	def __init__(self, n = 20, time_delay_from_t0 = 180,  num_of_interval = 6  ):
		"""
		Attribute 'self.accumulative_tuple_df' is a DataFrame used to contain the tuples which have been given to DecayPopularityModel for training. 
		type: pd.DataFrame

		Attribute 'self.event_type_strength' will convert an interaction event into a 
				   corresponding score that indicates how much the user likes the item.
		type: int/float/double

		Attribute 'self.time_delay_from_t0': t0 is the timepoint of an user-item interaction that is to be evaluated. 
				  The 'self.time_delay_from_t0' is the amount of time (in no. of days) before t0. When given an user-item interaction that
				  occurred at t0, the DecayPopularityModel will make recommendations based only on the user-item interactions 
				  that occurred within [t0 - self.time_delay_from_t0, t0]. Refer to the author's paper for more details.
		type: int 

		Attribute 'self.num_of_interval'. E.g. if self.num_of_interval == 6, then self.time_delay_from_t0 will be split into 6 time intervals, 
				   and each time interval will be given a different weight. E.g. if self.time_delay_from_t0 == 180 (i.e. 180 days or 6 months), and self.num_of_interval == 6,
				   Then, there will be 6 intervals, each consisting of 1 month. The more recent month will be given a higher weight compared to the earlier months. 
				   The exponential decay function is same as the function used by the author of the paper: e^(-tm)
				   Refer to the author's paper for more details.
		type: int


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

		self.time_delay_from_t0 = time_delay_from_t0 # i.e. by default, time_delay_from_t0 is set to 6 months (180 days)
		self.num_of_interval = num_of_interval # if num_of_interval == 6, then the self.time_delay_from_t0 will be divided into 6 time intervals, each consisting of 1 month (30 days).





	def update_model(self, train_tuple):

		if self.accumulative_tuple_df is None: # model is given its first tuple for training
			self.accumulative_tuple_df = pd.Dataframe(train_tuple).T
			self.accumulative_tuple_df.reset_index(inplace=True,drop=True)
		else:
			len_of_accumulative_tuple_df = self.accumulative_tuple_df.shape[0]
			self.accumulative_tuple_df.loc[len_of_accumulative_tuple_df] = train_tuple




	def rank_by_decay_popularity(self, row ): 

		"""
		param 'row': a series 
		type: pd.Series 

		A 'row' is an user-item interaction. This function will compute the decay popularity score (refer to the author's paper for formula) 
		for a 'row', sort it, and then return the top n most popular itemIDs. 
		Return the sorted np.array.

		return type: np.array

		"""

		temp_series_list = []
		for interval_index in range(self.num_of_interval):
			# interval_0 will contain itemIDs of user-item interactions that are 1 interval length away from t0, and interval 1 will contain itemIDs of interactions 2 interval lengths away and so on. 
			# Note that t0 refers to the timestamp of the user-item interaction in question.
			col_name = 'interval_' + str(interval_index) 

			weighted_count_series = row[col_name].value_counts(sort=True).apply(lambda x: x * np.exp(-(interval_index+1)) )
			temp_series_list.append(weighted_count_series)
		
		combined_df = pd.concat(temp_series_list, axis=1) # basically join all the series together by using itemID as key.
		summed_weighted_count_column = combined_df.sum(axis=1).sort_values(ascending=False)


		return summed_weighted_count_column.index.values[:self.n]   # return a sorted np.array containing the top n most popular itemIDs 


	def test_model(self, test_tuple_userID, test_tuple_timestamp):

		"""
		param 'test_tuple_userID': userID of the test tuple - userID is not relevant/useful in DecayPopularityModel 
		type: int

		param 'test_tuple_timestamp' is unix timestamp of the user-item interaction represented by the test tuple. 
		type: int 

		Return a dataframe containing the top n items that is recommended by the DecayPopularityModel.

		Return type: pd.Dataframe

		"""

		# if first tuple given to the model is a test tuple, then the DecayPopularityModel will not be able to make any recommendations
		if self.accumulative_tuple_df is None:
			return None 

		time_delay_from_t0_in_sec = self.time_delay_from_t0 * 60*60*24  # 60*60*24 is no. of seconds in a day.
		interval_length = round(time_delay_from_t0_in_sec/self.num_of_interval)

		prev_time_interval_start_point = test_tuple_timestamp
		relevant_tuples = pd.Series([])
		for interval_index in range(self.num_of_interval):
			col_name = 'interval_' + str(interval_index) 

			current_time_interval_start_point = test_tuple_timestamp - ( (interval_index+1) * interval_length)
			boolean_series_1 = self.accumulative_tuple_df['time'] >= current_time_interval_start_point
			boolean_series_2 = self.accumulative_tuple_df['time'] < prev_time_interval_start_point
			boolean_series_3 = boolean_series_1 & boolean_series_2

			relevant_tuples[col_name] = self.accumulative_tuple_df.loc[boolean_series_3, 'itemID'].copy() 
		
			prev_time_interval_start_point = current_time_interval_start_point

		top_n_recommendations = self.rank_by_decay_popularity(relevant_tuples)
		top_n_recommendations = pd.Series(top_n_recommendations)
		top_n_recommendations = pd.DataFrame( {'itemID': top_n_recommendations} )


		return top_n_recommendations

	def train_in_batch(self, dataset):
		if self.accumulative_tuple_df is not None:
			print("Note: 'train_in_batch()' is called, and the data samples given to decay_popularity_model for training have been replaced")
		
		self.accumulative_tuple_df = dataset.copy()





"""
	# this method for illustration only, decay_popularity_model is not a model that learn or predict by batches of data, so this method is incorrect by definition.
	def compute_most_popular_in_batch(self, dataset):
		\"""
		param 'dataset' can either be the training set or the validation set.
		type: pd.DataFrame

		Will compute a new column in 'dataset' that is named 'top_n_most_popular'. 
		For each row in 'dataset', the 'top_n_most_popular' column contains the top n items that have the highest decay popularity score.

		Return the modified 'dataset'

		Return type: pd.DataFrame

		\"""

		time_delay_from_t0_in_sec = self.time_delay_from_t0 * 60*60*24  # 60*60*24 is no. of seconds in a day.
		interval_length = round(time_delay_from_t0_in_sec/self.num_of_interval)

		dataset.reset_index(inplace=True) # creates an 'index' column

		# For each user-item interaction, identify the previous user-item interactions that are relevant (i.e. falls within the interval)
		total_column_indices_list = [ [] for i in range(self.num_of_interval) ]
		for index, row in dataset.iterrows():
			prev_col_itemlist_len = 0
			for interval_index in range(self.num_of_interval):
				# interval_0 will contain itemIDs of user-item interactions that are 1 interval length away from t0, and interval 1 will contain itemIDs of interactions 2 interval lengths away and so on. 
				# Note that t0 refers to the timestamp of the user-item interaction in question.
				col_name = 'interval_' + str(interval_index) 

				boolean_series = dataset['time'] >= row['time'] - ( (interval_index+1) * interval_length)
				boolean_series[index-prev_col_itemlist_len:] = False
				indices_list = dataset.loc[boolean_series, 'index'].tolist()
				prev_col_itemlist_len = len(indices_list)
				total_column_indices_list[interval_index].append(indices_list)

		itemID_column = dataset['itemID']
		for interval_index in range(self.num_of_interval):
			col_name = 'interval_' + str(interval_index) 
			dataset[col_name] = pd.Series(total_column_indices_list[interval_index]).apply(lambda x: itemID_column[x] )

		dataset['top_n_most_popular'] = dataset.apply(func = self.rank_by_decay_popularity, axis = 1)			

		return dataset

	# this method for illustration only, decay_popularity_model is not a model that learn or predict by batches of data, so this method is incorrect by definition.
	def train_in_batch(self, dataset):
		\"""
		param 'dataset' is the data samples used for training of the model.
		type: pd.DataFrame

		This method is optional as a decay popularity model does not need to be trained.
		For each user-item interaction in 'dataset', this method will predict the top n most popular item for it.
		
		By convention, this inherited method 'train_in_batch' is not designed to return an object. 
		But decay popularity model will return 'dataset' with an additional column 'top_n_most_popular' that, 
		for each row in 'dataset', contains the top n items that have the highest decay popularity score.
		This return object can be left unused if it is not needed.

		Return type: pd.DataFrame

		\"""

		modified_dataset = self.compute_most_popular_in_batch(dataset.copy())
		return modified_training_set

	# this method for illustration only, decay_popularity_model is not a model that learn or predict by batches of data, so this method is incorrect by definition.
	def predict_in_batch(self, validation_set):
		\"""
		param 'validation_set' is the data samples used for validation of the model.
		type: pd.DataFrame

		The decay popularity model is not designed to predict in batch (instead, it should be used for prediction in streaming environment.)
		Here, we provide a rudimentary way to force the decay popularity model to predict in batch. More details below.
		For each user in 'validation_set', predict the top n item recommended by the model. The decay popularity model would use the top n most popular list 
		given to every user's last interaction.
		Return a dataframe containing the recommendations for each user. The dataframe adheres to the following format:
			The dataframe will have 3 columns: 1. 'userID' 2. 'itemID_list', and 3.'action_list'
			The 'itemID_list' column contains a list of recommended itemID for each user.
			The 'action_list' column contains a list of recommended action for each user. Each action must be an enum defined in a subclass of dataset.Action 
		
		return type: pd.DataFrame


		\"""

		modified_validation_set = self.compute_most_popular_in_batch(validation_set.copy())

		userID_list = modified_validation_set['userID'].unique()
		recommendation_df = pd.DataFrame(userID_list, columns=['userID'])

		recommendation_df['itemID_list'] = modified_validation_set.groupby('userID').apply(lambda x: x.iloc[-1])['top_n_most_popular']
		recommendation_df['action_list'] = None	 # no value is needed to be given to action_list if prediction of action is not required.

		return recommendation_df
"""




