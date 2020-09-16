# RecEval-Evaluation-Framework

## Usage

To illustate, we would show how to evaluate a popularity model with the Deskdrop article sharing dataset from CI&T using RecEval. The entire process composes of the following steps:  
- Create a class Deskdrop_Dataset that inherits from the abstract class Dataset. In Deskdrop_Dataset, the following needs to be done:
  -  Define an inner class, Action, that is simply an Enum class containing all possible actions between an user and an item. For instance, the Action can have the following Enum members: VIEW = 1, LIKE = 2 , BOOKMARK = 3 , FOLLOW = 4, COMMENT = 5. 
  -  Create an attribute, time_arranged_interactions_df, that a pd.DataFrame containing user-item interactions where each row is a pd.Series of the following format: (time, userID, itemID, action, data_split_label). It is ordered chronologically such that topmost row is the earliest interaction.  

- Create a class PopularityModel that inherits from the abstract class Model. In PopularityModel, the following needs to be done:
  -  Implement the abstract method, test_model(test_tuple_userID, test_tuple_timestamp), that would return a dataframe containing the top n items that is recommended for a user by the popularity model. For the popularity model, the top n items would be the n most popular items for every user.
  -  Optionally, the abstract method, train_in_batch(training_set), can be implemented to allow the popularity model to train using a batch of data samples. It is equally appropriate for programmers using RecEval to define their own training method. For the popularity model, the train_in_batch method would need to compute and store a list of items sorted by popularity.
  -  Also, optionally, the abstract method, predict_in_batch(validation_set), can be implemented if hyperparameter tuning using the validation set needs to be done for the model. The method would, for each user in validation_set, predict the top n item recommended by the model.  For the popularity model, this method would just predict the top n most popular items for every user.
  -  Lastly, the optional abstract method, update_model(train_tuple), needs to be implemented if the popularity model is to be updated after observing each train tuple during the sequential phase. If the model does not update incrementally, the method can be left blank by placing a pass keyword. For the popularity model, this method would re-compute and update the list of items sorted by popularity.

- Modify a template main.py script and run it. The main.py script will do the following:
  -  Initialize the Deskdrop_Dataset, PopularityModel, and Evaluator objects.
  -  Retrieve the training set from Deskdrop_Dataset and use it to train the PopularityModel.
  -  If needed, do validation of PopularityModel using the validation set. Otherwise the validation set will be used to train the model as well.
  -  In the sequential phase, evaluate the PopularityModel with the testing set and report the metrics scores.


## Available Algorithms
We provide a few recommendation algorithms to function as benchmarks:
- Popularity Model: Based on all the tuples have been fed into the model for training, recommend the top N most popular items. Popularity of an item is computed based on its interaction count.
- Decay Popularity Model: It uses an exponential weighted sliding window to compute the top N most popular items for each test tuple. The top N most popular items recommended for a test tuple would have the most user-item interactions within the window after factoring time-weighting, which gives older interactions exponentially less weight.
- ItemKNN Model
- Singular Value Decomposition (SVD) Model
- NMF Model 


## Available Datasets
RecEval contains a few popular recommendation datasets used for convenience:
- MovieLens Dataset 
- CI\&T Deskdrop Dataset (https://www.kaggle.com/gspmoreira/articles-sharing-reading-from-cit-deskdrop)
- LastFM Dataset (http://www.lastfm.com)
- Amazon Automotive Dataset (http://jmcauley.ucsd.edu/data/amazon)
- Delicious Social Bookmarking Dataset (http://www.delicious.com)
