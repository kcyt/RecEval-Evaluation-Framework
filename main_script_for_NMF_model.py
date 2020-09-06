import pandas as pd
from dataset import Deskdrop_Dataset
from NMF_model import NMF_Model
from evaluator import Evaluator

"""
An example on using this evaluation framework: You can replace 'deskdrop_dataset' and 'model_NMF' with your own dataset and model objects. 
But your dataset object must be a subclass of Dataset defined in dataset.py and implement its abstract methods; 
and your model object must be a subclass of Model defined in model.py and implement its abstract methods.
"""

# tunable parameters
num_of_recommendations = 5 # number of recommendations to be given by the model.
n_ranks=[1, 3, 5] # For defining the metrics e.g. if n_ranks = [1,5], and the metrics used is RECALL, then RECALL@1 and RECALL@5 will be used

# initialising the objects
deskdrop_dataset = Deskdrop_Dataset(partition_ratios = [0.6, 0.2, 0.2]) 
model_NMF = NMF_Model(n =num_of_recommendations )
evaluator = Evaluator(n_ranks=n_ranks)

# do Training 
training_set = deskdrop_dataset.getTraining()
model_NMF.train_in_batch(dataset=training_set)

# Validation in stream environment 
deskdrop_dataset.set_curr_pointer(mode='Validation')
validation_results = evaluator.evaluate_model_in_stream( dataset=deskdrop_dataset, model=model_NMF, mode='Validation', scheme = [0.9,0.1])

# do Training with both Training and Validation sets:
validation_set = deskdrop_dataset.getValidation(exclude_first_time_users = False)
train_and_validate_sets = pd.concat([training_set, validation_set])
model_NMF.train_in_batch(dataset=train_and_validate_sets)

# evaluate your model
deskdrop_dataset.set_curr_pointer(mode='Testing')
testing_results = evaluator.evaluate_model_in_stream( dataset=deskdrop_dataset, model=model_NMF, mode='Testing', scheme = [0.9,0.1])

