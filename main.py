from dataset import Deskdrop_Dataset
from model import PopularityModel
from evaluator import Evaluator

num_of_tuples_to_use = 100
num_of_recommendations = 5 # number of recommendations to be given by the model.
n_ranks=[1, 3, 5] # For defining the metrics e.g. if n_ranks = [1,5], and the metrics used is RECALL, then RECALL@1 and RECALL@5 will be used

deskdrop_dataset = Deskdrop_Dataset(test_data_proportion = 0.2, folder_path= 'CI&T_Deskdrop_dataset')
popularity_model = PopularityModel(n=num_of_recommendations, popularity_metric = "simple interaction count")
evaluator = Evaluator(n_ranks=n_ranks)
"""

'tuple' is a row of a Dataframe i.e. a Series
'list_of_recommendations' is a Dataframe of length n, where each row represents a recommended item. The 1st row should be the top item that the model would like to recommend.

"""

for iteration_no in range(num_of_tuples_to_use):
	tuple = deskdrop_dataset.getNextTuple()
	if tuple['train_test_label'] == 'Train':
		popularity_model.update_model(tuple)
	elif tuple['train_test_label'] == 'Test':
		list_of_recommendations = popularity_model.test_model(tuple['userID'])
		evaluator.evaluateList(list_of_recommendations, tuple)
	else:
		raise NotImplementedError

evaluator.print_results()