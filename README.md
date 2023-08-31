# CommonLit - Evaluate Student Summaries
Aim to build a model that evaluates how well a student represents the main idea and details of a source text, as well as the clarity, precision, and fluency of the language used in the summary.

In our project, we will try to use several method to build some modles and find the best one.

[Competition from Kaggle](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries)


## Code List 
- pre_processing_part1.py
- pre_processing_part2.py
- score_distribution.py
- support_vector.py
- kernal_ridge.py
- knn.py
- random_forest.py
- xgboost_regressor.py
- neural_network.py

## Dataset List
- summaries_train.csv
- summaries_test.csv
- prompts_train.csv
- prompts_test.csv
- sample_submission.csv

## Environment
Before you run the project, you need to install some packages.

| Name       | Description                                                                                                                                        |
|------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| numpy      | a popular Python library for numerical computing                                                                                                   |
| pandas     | a powerful Python library widely used for data manipulation and analysis                                                                           |
| nltk       | a powerful Python library for natural language processing (NLP)                                                                                    |
| sklearn    | a popular and widely used Python library for machine learning                                                                                      |
| xgboost    | an optimized and scalable open-source machine learning library designed for gradient boosting                                                      |
| tensorflow | an open-source deep learning framework developed by the Google Brain team.<br/>Keras has also been integrated as a high-level API into TensorFlow. |

```python
pip install numpy
pip install pandas
pip install nltk
pip install sklearn
pip install xgbo
pip install tensorflow
```

Then, you need to make sure that all files(in Code and Dataset list) are in the same directory.

## Run the project
**Run the project on windows:**

Step1. <br/>Open the cmd in the directory.

Step2. <br/>Clean the natural language in summaries_train.csv and prompts.csv. <br/>Then, you will get clean_data.csv.
```python
python pre_processing_part1.py
```


Step3. <br/>Split the clean_data.csv. Divide the data into new training and testing sets.
<br/>Calculate the TF-IDF and aim to convert a collection of text documents into numerical feature vectors, enabling machine learning algorithms to work with text data.
<br/>Then you will get xtrain.csv , ytrain.csv , xtest.csv and ytest.csv.
```python
python pre_processing_part2.py
```

Step4. <br/>Using different method to build models, by running the different python files
```python
python support_vector.py
python kernal_ridge.py
python knn.py
python random_forest.py
python xgboost_regressor.py
python neural_network.py
```

Additional Step. <br/> Output the distribution of Scores in dataset and it doesn't affect the model training process. 
```python
python score_distribution.py
```