from dataset import Deskdrop_Dataset
from model import PopularityModel
from evaluator import Evaluator

"""
An example on using this evaluation framework: You can replace 'deskdrop_dataset' and 'popularity_model' with your own dataset and model objects. 
But your dataset object must be a subclass of Dataset defined in dataset.py and implement its abstract methods; 
and your model object must be a subclass of Model defined in model.py and implement its abstract methods.
"""


# tunable parameters
num_of_recommendations = 5 # number of recommendations to be given by the model.
n_ranks=[1, 3, 5] # For defining the metrics e.g. if n_ranks = [1,5], and the metrics used is RECALL, then RECALL@1 and RECALL@5 will be used

# initialising the objects
deskdrop_dataset = Deskdrop_Dataset(partition_ratios = [0.4, 0.2, 0.4], folder_path= 'CI&T_Deskdrop_dataset') 
popularity_model = PopularityModel(n=num_of_recommendations, popularity_metric = "simple interaction count")
evaluator = Evaluator(n_ranks=n_ranks)

# do Training 
training_set = deskdrop_dataset.getTraining()
popularity_model.train_in_batch(training_set=training_set)

# do Validation
validation_set = deskdrop_dataset.getValidation(exclude_first_time_users = False)
recommendation_df = popularity_model.predict_in_batch(validation_set=validation_set)
validation_results = evaluator.evaluate_Validation_Recommendations(recommendation_df, validation_set )


# evaluate your model
evaluation_results = evaluator.evaluateModel( dataset=deskdrop_dataset, model=popularity_model, scheme = [0.1,0.9])