# Part 0. Data Exploration and Cleaning

# Spark session initiation
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline
import os
os.environ["PYSPARK_PYTHON"] = "python3"

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("youtube comment analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# load the dataset
df_clean=spark.read.load("/FileStore/tables/animals_comments.csv", format='csv', header = True)
df_clean.show(10)

# look at the total number of rows in the dataset
print("Total number of rows is: ", df_clean.count())

# look at the total number of non-null rows in the dataset
df_clean = df_clean.na.drop(subset=["comment"])
print("Number of non-null rows in the data is: ", df_clean.count())
df_clean.show()

# find user with preference of dog and cat
from pyspark.sql.functions import when
from pyspark.sql.functions import col

# extract the comment with certain key words to generate the new binary-label column
df_clean = df_clean.withColumn("label", \
                           (when(col("comment").like("%dog%"), 1) \
                           .when(col("comment").like("%I have a dog%"), 1) \
                           .when(col("comment").like("%cat%"), 1) \
                           .when(col("comment").like("%I have a cat%"), 1) \
                           .when(col("comment").like("%my puppy%"), 1) \
                           .when(col("comment").like("%my pup%"), 1) \
                           .when(col("comment").like("%my kitty%"), 1) \
                           .otherwise(0)))


# show the preprocessed data with indexed label
df_clean.show()

# Part 1. Data preprocessing and Build the classifier
from pyspark.ml.feature import RegexTokenizer, Word2Vec
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# regular expression tokenizer
regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="words", pattern="\\W")

# word2vec model setup
word2Vec = Word2Vec(inputCol="words", outputCol="result")

from pyspark.ml import Pipeline

# set the pipeline with two stages: regexTokenizer and word2Vec
pipeline = Pipeline(stages=[regexTokenizer, word2Vec])

# Fit the pipeline to training documents.
pipelineFit = pipeline.fit(df_clean)
dataset = pipelineFit.transform(df_clean)

# show the trained dataset
dataset.show()

# split the original dataset into train and test set stratified with label.
(lable0_train,lable0_test)=dataset.filter(col('label')==1).randomSplit([0.7, 0.3],seed = 100)
(lable1_train, lable1_ex)=dataset.filter(col('label')==0).randomSplit([0.003, 0.997],seed = 100)
(lable1_test, lable1_ex2)=lable1_ex.randomSplit([0.001, 0.999],seed = 100)

trainingData = lable0_train.union(lable1_train)
testData = lable0_test.union(lable1_test)

print("Dataset Count: " + str(dataset.count()))
print("Training Dataset Count: " + str(trainingData.count()))
print("Test Dataset Count: " + str(testData.count()))


# Model 1. Logistic Regression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# set up the logistic regression model
lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol = 'result',labelCol='label')

# construct a grid of parameters to search over, 3x2 = 6 models in this case
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.01]) \
    .build()

# 5-fold cross-validation
cv = CrossValidator(estimator=lr,
                   estimatorParamMaps=paramGrid,
                   evaluator=BinaryClassificationEvaluator(),
                   numFolds=5)

# run cross-validation, and choose the best set of parameters
cvModel = cv.fit(trainingData)

# model selection and application on test data
lr_best_model = cvModel.bestModel

lr_predictions=lr_best_model.transform(testData)

# calculate the accuracy of the model
evaluator=BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='rawPrediction',metricName='areaUnderROC')
AUC = evaluator.evaluate(lr_predictions)

TP = lr_predictions[(lr_predictions["label"] == 1) & (lr_predictions["prediction"] == 1.0)].count()
FP = lr_predictions[(lr_predictions["label"] == 0) & (lr_predictions["prediction"] == 1.0)].count()
TN = lr_predictions[(lr_predictions["label"] == 0) & (lr_predictions["prediction"] == 0.0)].count()
FN = lr_predictions[(lr_predictions["label"] == 1) & (lr_predictions["prediction"] == 0.0)].count()

accuracy = (TP + TN)*1.0 / (TP + FP + TN + FN)
precision = TP*1.0 / (TP + FP)
recall = TP*1.0 / (TP + FN)

print("Prediction result summary for Logistic Regression Model:  ")

print ("True Positives:", TP)
print ("False Positives:", FP)
print ("True Negatives:", TN)
print ("False Negatives:", FN)
print ("Test Accuracy:", accuracy)
print ("Test Precision:", precision)
print ("Test Recall:", recall)
print ("Test AUC of ROC:", AUC)

# extract best logistic regression parameters
print("Logistic Regression best maxIter: ", lr_best_model._java_obj.parent().getMaxIter())

print("Logistic Regression best regParam: ", lr_best_model._java_obj.parent().getRegParam())

print("Logistic Regression best elasticNetParam: ", lr_best_model._java_obj.parent().getElasticNetParam())

lr_predictions.printSchema()
lr_predictions.select("label","probability","prediction").show()

# plot the ROC curve
trainingSummary = lr_best_model.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# plot the precision-recall tradeoff curve
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# Model 2. Random forest
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer

# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="result", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="result", numTrees=10)

pipeline = Pipeline(stages=[featureIndexer, rf])

# train the pipeline with training data
rfModel = pipeline.fit(trainingData)

# make predictions on the test data using the random forest model
RFpredictions = rfModel.transform(testData)
RFpredictions.printSchema()

# evaluate the random forest classifier by area under ROC
evaluator = BinaryClassificationEvaluator()
AUC = evaluator.evaluate(RFpredictions, {evaluator.metricName: "areaUnderROC"})

TP = RFpredictions[(RFpredictions["label"] == 1) & (RFpredictions["prediction"] == 1.0)].count()
FP = RFpredictions[(RFpredictions["label"] == 0) & (RFpredictions["prediction"] == 1.0)].count()
TN = RFpredictions[(RFpredictions["label"] == 0) & (RFpredictions["prediction"] == 0.0)].count()
FN = RFpredictions[(RFpredictions["label"] == 1) & (RFpredictions["prediction"] == 0.0)].count()

accuracy = (TP + TN)*1.0 / (TP + FP + TN + FN)
precision = TP*1.0 / (TP + FP)
recall = TP*1.0 / (TP + FN)

print("Prediction result summary for Random Forest Model:  ")

print ("True Positives:", TP)
print ("False Positives:", FP)
print ("True Negatives:", TN)
print ("False Negatives:", FN)
print ("Test Accuracy:", accuracy)
print ("Test Precision:", precision)
print ("Test Recall:", recall)
print("Test Area Under ROC: ", AUC)

# summary of the random forest model
randomFModel = rfModel.stages[1]
print(randomFModel)

trainingSummary = randomFModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# plot the precision-recall tradeoff curve
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()

# Model 3. Gradient boosting

from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
    VectorIndexer(inputCol="result", outputCol="indexedFeatures", maxCategories=4).fit(trainingData)

gbt = GBTClassifier(maxIter=10, featuresCol = "result", labelCol = "label")
pipeline = Pipeline(stages=[featureIndexer, gbt])


# model tuning and cross validation
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [2, 4, 6])
             .addGrid(gbt.maxBins, [20, 60])
             .addGrid(gbt.maxIter, [10, 20])
             .build())

cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=BinaryClassificationEvaluator(), numFolds=5)

# run cross-validation, and choose the best set of parameters
# train the pipeline with training data
gbtModel = cv.fit(trainingData)

gbt_best_model = gbtModel.bestModel
gbt_predictions = gbt_best_model.transform(testData)

evaluator=BinaryClassificationEvaluator(
          labelCol='label', rawPredictionCol='rawPrediction',metricName='areaUnderROC')
AUC = evaluator.evaluate(gbt_predictions)

TP = gbt_predictions[(gbt_predictions["label"] == 1) & (gbt_predictions["prediction"] == 1.0)].count()
FP = gbt_predictions[(gbt_predictions["label"] == 0) & (gbt_predictions["prediction"] == 1.0)].count()
TN = gbt_predictions[(gbt_predictions["label"] == 0) & (gbt_predictions["prediction"] == 0.0)].count()
FN = gbt_predictions[(gbt_predictions["label"] == 1) & (gbt_predictions["prediction"] == 0.0)].count()

accuracy = (TP + TN)*1.0 / (TP + FP + TN + FN)
precision = TP*1.0 / (TP + FP)
recall = TP*1.0 / (TP + FN)

print("Prediction result summary for Gradient boosting Model:  ")

print ("True Positives:", TP)
print ("False Positives:", FP)
print ("True Negatives:", TN)
print ("False Negatives:", FN)
print ("Test Accuracy:", accuracy)
print ("Test Precision:", precision)
print ("Test Recall:", recall)
print("Test Area Under ROC: ", AUC)

# summary of the random forest model
gradientBModel = gbt_best_model.stages[1]
print(gradientBModel)

trainingSummary = gradientBModel.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# plot the precision-recall tradeoff curve
pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()


# The best model is then determined to be logistic regression based on the 
# evaluation of accuracy, precision, recall rates as well as the area under ROC. 
# Now the logistic regression model is used to gain insights in the cat and dog owners.

# Part 2. Classify All The Users with the best model

prediction_all = lr_best_model.transform(dataset)
prediction_all.show(10)

# Part 3. Get insigts of Users
import pyspark.sql.functions as F
#number of total user
total_user = prediction_all.select('userid').distinct().count()

# number of owner predicted
owner_pred = prediction_all.select('userid').distinct().filter(prediction_all['prediction']==1.0).count()

# predicted fraction of users who are cat/dog owners
fraction = owner_pred/total_user
print("Total number of users: ", total_user)
print("Number of predicted owner: ", owner_pred)
print('Fraction of the users who are cat/dog owners (ML estimate): ', round(fraction,3))

from pyspark.ml.feature import StopWordsRemover
import matplotlib.cm as cm
from matplotlib import rcParams
%matplotlib inline

df_all_owner = dataset.select('words').union(prediction_all.filter(F.col('prediction') == 1.0).select('words'))

stopwords_custom = ['im', 'get', 'got', 'one', 'hes', 'shes', 'dog', 'dogs', 'cats', 'cat', 'kitty', 'much', 'really', 'love','like','dont','know','want','thin',\
                    'see','also','never','go','ive']

remover1 = StopWordsRemover(inputCol="raw", outputCol="filtered")
core = remover1.getStopWords()
core = core + stopwords_custom
remover = StopWordsRemover(inputCol="words", outputCol="filtered",stopWords=core)
df_all_owner = remover.transform(df_all_owner)

wc = df_all_owner.select('filtered').rdd.flatMap(lambda a: a.filtered).countByValue()

df_all_owner.show(10)

# wcSorted = wc.sort(lambda a: a[1])
wcSorted = sorted(wc.items(), key=lambda kv: kv[1],reverse = True)
wcSorted

# Part 4. Identify Creators With Cat And Dog Owners In The Audience
# Get all creators whenever the users label is True(cat/dog owner)
df_create = dataset.select('creator_name').union(prediction_all.filter(F.col('prediction') == 1.0).select('creator_name'))

df_create.createOrReplaceTempView("create_table")

# get count
create_count = spark.sql("select distinct creator_name, count(*) as Number\
                          from create_table \
                          group by creator_name \
                          order by Number DESC")

create_count.show()