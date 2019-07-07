We provide 3 data set for the task.

Data set 1: Ratings for training
- File: case_study_reco_ratings_train.csv
- Rows: A rating given to an item by a user in at a specific date
- Columns:
	- user_id: Id of the user
	- item_id: Id of the item
	- rating: Value of the rating
	- p: Flag whether the rating is higher than 8 (is the rating "very good"?)

Data set 2: Target set of user-item pairs
- File: case_study_reco_ratings_target.csv
- Rows: A target user-item pair for the prediction should be made
- Columns:
	- user_id: Id of the user
	- item_id: Id of the item
	- p: Target variable that should be estimated: The likelihood of being rated "very good" by the user (See case_study_reco_ratings_target_example.csv)

Data set 3: Item-category assignment
- File: case_study_reco_item_category.csv
- Rows: Each row represents whether the item belongs to a category. (One item can be assigned to multiple categories)
- Columns:
	- item_id: Id of the item
	- category: Id of the item category