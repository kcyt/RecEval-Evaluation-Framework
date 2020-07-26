import numpy as np
import pandas as pd 


class Model:

	def __init__(self, n = 20):

		"""
		param 'n' refers to the top n recommendations to be returned by the Model
		type: int

		"""

		self.n = n 
 

	# Abstract method
	def update_model(self, train_tuple):
		"""
		param 'train_tuple': is the tuple given to the model for training purpose.
							 format of a tuple: <time, userID, itemID, action, data_split_label>
		type: pd.Series object

		Train/Fine-tune the model given 1 train tuple

		No return object.


		"""

		raise NotImplementedError("Please override and implement the 'update_model(self, train_tuple)' method in your model")


	# Abstract method
	def test_model(self, test_tuple_userID, test_tuple_timestamp):
		"""
		param 'test_tuple_userID' is ID of the user corresponding to the test tuple.
		type: int

		param 'test_tuple_timestamp' is unix timestamp of the user-item interaction represented by the test tuple. 
		type: int 

		Retrieve a ranked list of items based only on the userID and timestamp of the test_tuple.
		Concretely, will return a pd.DataFrame object containing the top n items that is recommended by the popularity model. 
		The 1st item to be recommended should be at the top of the DataFrame object, and the DataFrame must have at least a 'itemID' column. Additional/Other columns are optional.
		
		return type: pd.DataFrame object 

		"""
		
		raise NotImplementedError("Please override and implement the 'test_model(self, test_tuple_userID)' method in your model")



	# Optional Abstract method (it is equally appropriate to call a custom function that trains the model.)
	def train_in_batch(self, dataset):
		"""
		param 'dataset' is the data samples used for training of the model. (e.g. training set, or concatentation of training set and validation set)
		type: pd.DataFrame

		To train the model using 'dataset'.
		
		No return object.

		"""
		
		raise NotImplementedError("The optional train_in_batch() has not been implemented and thus cannot be used.")



	# Optional Abstract method (required only if validation of the model is needed.)
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
		
		raise NotImplementedError("The optional predict_in_batch() has not been implemented and thus cannot be used.")






