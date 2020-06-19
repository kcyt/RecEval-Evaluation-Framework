from dataset import Deskdrop_Dataset
from model import PopularityModel
from evaluator import Evaluator

"""
An example on using this evaluation framework: You can replace 'deskdrop_dataset' and 'popularity_model' with your own dataset and model objects. 
But your dataset object must be a subclass of Dataset defined in dataset.py and implement its abstract methods; 
and your model object must be a subclass of Model defined in model.py and implement its abstract methods.
"""


# tunable parameters
num_of_tuples_to_use = 1000 # number of tuples (user-item interactions) to be used for training and testing the model.
num_of_recommendations = 5 # number of recommendations to be given by the model.
n_ranks=[1, 3, 5] # For defining the metrics e.g. if n_ranks = [1,5], and the metrics used is RECALL, then RECALL@1 and RECALL@5 will be used

# initialising the objects
deskdrop_dataset = Deskdrop_Dataset(test_data_proportion = 0.2, kfold_num_splits = 5, folder_path= 'CI&T_Deskdrop_dataset') # set kfold_num_splits to 0 or 1 to disable kfold.
popularity_model = PopularityModel(n=num_of_recommendations, popularity_metric = "simple interaction count")
evaluator = Evaluator(n_ranks=n_ranks)

# evaluate your model
evaluation_results = evaluator.evaluateModel( dataset=deskdrop_dataset, model=popularity_model, num_of_tuples_to_use = num_of_tuples_to_use, use_cross_validate = True)