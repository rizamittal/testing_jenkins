import time
from datetime import datetime
from pyspark.sql.functions import max, first
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, HiveContext
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import time
import torch.nn as nn
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import sys
from pyspark.sql.window import Window
import re

# create spark session
appName = "consumer twitter"
master = "local[6]"

spark = SparkSession.builder \
    .master(master) \
    .appName(appName) \
    .enableHiveSupport() \
    .getOrCreate()

# get stop words
stop_words = ['a', 'the', 'rishi', 'sunak', 'and', 'was', 'ers', 'which', 'how', 
                'who', 'his', 'not', 'didn', 'from', 'has', 'there', 'he', 'them',
                'here', 'had', 'did', 'about', 'going', 'only', 'too', 'also', 
                'next', 'with', 'its', 'she', 'her', 'we', 'it', 'these', 'then',
                'your', 'himself']

# load in BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BERT_encoder = BertModel.from_pretrained("bert-base-uncased")

# define model architecture
class BERT_Classifier(nn.Module):
    def __init__(self):
        super(BERT_Classifier, self).__init__()
        self.fc1 = nn.Linear(768, 768)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(768, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.layernorm = nn.LayerNorm(128)
    def forward(self, x):
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.layernorm(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# load in model
classifier_head = torch.load('BERT_classifier_head')


# format df function for dashbord utility
def create_df(df_filtered):

    # decode tweets and put in list
    tweet_list = df_filtered.rdd.map(lambda x:x['value'].decode()).collect()

    # put encoded tweets into torch array
    encoded_tweet_array = torch.zeros((len(tweet_list), 768))
    print(encoded_tweet_array.shape)
    for i, tweet in enumerate(tweet_list):
            encoded_tweet_array[i, :] = torch.reshape(BERT_encoder(**tokenizer(tweet, return_tensors='pt'))[1].detach(), (1, 768))

    # get probabilities for each tweet using classifier head
    with torch.no_grad():
            probs_df = spark.createDataFrame(pd.DataFrame(classifier_head.eval().forward(encoded_tweet_array).numpy(), columns=['probability']))
    
    # get common column for join
    probs_df2 = probs_df.withColumn("id", row_number().over(Window.orderBy(lit(1))))
    value_df2 = df_filtered.withColumn("id", row_number().over(Window.orderBy(lit(1))))

    # join dfs
    joined_df = value_df2.join(probs_df2, "id").drop(col("id"))
    
    # create prediction string column
    pred_udf = udf(lambda x: "positive" if x >= 0.6 else ("negative" if x <= 0.4 else "neutral"), StringType())
    complete_df = joined_df.withColumn('prediction', pred_udf(joined_df.probability))

    complete_df2 = complete_df.withColumn("value", col("value").cast(StringType())).drop(col("key"))
    return complete_df2


# performs regex operations for word cloud in dashboard
@udf(returnType=StringType())
def perform_regex(row):
	if row != None:
		Tweet_Texts_Cleaned = row.lower()
		# Removing the twitter usernames from tweet string
		Tweet_Texts_Cleaned=re.sub(r'@\w+', ' ', Tweet_Texts_Cleaned)
		# Removing the URLS from the tweet string
		Tweet_Texts_Cleaned=re.sub(r'http\S+', ' ', Tweet_Texts_Cleaned)
		# Deleting everything which is not characters
		Tweet_Texts_Cleaned = re.sub(r'[^a-z A-Z]', ' ',Tweet_Texts_Cleaned)
		# Deleting any word which is less than 3-characters mostly those are stopwords
		Tweet_Texts_Cleaned= re.sub(r'\b\w{1,2}\b', '', Tweet_Texts_Cleaned)
		# Stripping extra spaces in the text
		Tweet_Texts_Cleaned= re.sub(r' +', ' ', Tweet_Texts_Cleaned)
	return Tweet_Texts_Cleaned


# turn stripped tweets col into single string
def convert_to_string(df, stop_words):
        df2 = df.withColumn("value", col("value").cast(StringType()))
        df3 = df2.withColumn("stripped_tweets", perform_regex(col("value")))
        list_tweets = df3.select("stripped_tweets").rdd.flatMap(lambda x: x).collect()
        string_tweets = "".join(list_tweets)
        word_list = string_tweets.split(" ")
        sw_removed = [x for x in word_list if x not in stop_words]
        return sw_removed


# get tweets per second
def tweet_rate_calc(df):
        tweets_per_sec = df.count() / 15
        time = df.first()['timestamp']
        rate_list = [(time, tweets_per_sec)]
        rate_df = spark.createDataFrame(rate_list, schema=['timestamp', 'num_tweets'])
        return rate_df


# define variables for loop
x = 0
kafka_servers = "ip-172-31-3-80.eu-west-2.compute.internal:9092, ip-172-31-13-101.eu-west-2.compute.internal:9092, ip-172-31-5-217.eu-west-2.compute.internal:9092, ip-172-31-9-237.eu-west-2.compute.internal:9092"

# enter endless loop
while x == 0:

        # read data from kafka broker
        df = spark \
        .read \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_servers) \
            .option("subscribe", "twitter_version_1") \
        .load()

        # check if any messages have been sent to hive
        if df.count() > 0:
                # check if raw stock data table exists in hive and get offset if it does
                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'twitter_data'):
                        # filter kafka message by max offset
                        max_offset = spark.sql("select MAX(offset) from curated_layer_db.twitter_data")
                        df_filtered = df.filter(df.offset > max_offset.first()['max(offset)'])

			# create word cloud string #
                        word_list = convert_to_string(df_filtered, stop_words)

                        # get tweets per second
                        tweet_rate_df = tweet_rate_calc(df_filtered)
			
                        if df_filtered.count() > 0: # only append if new messages exist
                                time_of_msg = time.time()
                                df_filtered.write.mode('append').format('hive').saveAsTable('raw_layer_db.twitter_data')
                                formatted_df = create_df(df_filtered) # create formatted df

                                #### check main formatted twitter data exists ####
                                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'twitter_data'):
					
                                        formatted_df.write.mode('append').format('hive').saveAsTable('curated_layer_db.twitter_data')
                                else:
                                        formatted_df.write.mode('overwrite').format('hive').saveAsTable('curated_layer_db.twitter_data')
				
				#### check wordcloud dataframe exists ####
                                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'word_cloud'):
                                        word_cloud_df = spark.sparkContext.parallelize(word_list).toDF("string")
                                        word_cloud_df.write.mode('append').format('hive').saveAsTable('curated_layer_db.word_cloud')
                                else:
                                        word_cloud_df = spark.sparkContext.parallelize(word_list).toDF("string")
                                        word_cloud_df.write.mode('overwrite').format('hive').saveAsTable('curated_layer_db.word_cloud')
                                
                                #### check tweet rate dataframe exists ####
                                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'word_cloud'):
                                        tweet_rate_df.write.mode('append').format('hive').saveAsTable('curated_layer_db.tweet_rate')
                                else:
                                        tweet_rate_df.write.mode('overwrite').format('hive').saveAsTable('curated_layer_db.tweet_rate')



                else:
                        max_offset = -1 # first offset will be 0 so -1 means we will take that first output
                        if df.filter(df.offset > max_offset).count() > 0:
                                time_of_msg = time.time()
                                df.write.mode('overwrite').format('hive').saveAsTable('raw_layer_db.twitter_data')

				# create word cloud string #
                                word_list = convert_to_string(df, stop_words)

                                formatted_df = create_df(df) # create formatted df

                                # get tweets per second
                                tweet_rate_df = tweet_rate_calc(df)

                                ### check main formatted dataframe exists
                                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'twitter_data'):
                                        formatted_df.write.mode('append').format('hive').saveAsTable('curated_layer_db.twitter_data')
                                else:
                                        formatted_df.write.mode('overwrite').format('hive').saveAsTable('curated_layer_db.twitter_data')

				#### check wordcloud dataframe exists ####
                                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'word_cloud'):
                                        word_cloud_df = spark.sparkContext.parallelize(word_list).toDF("string")
                                        word_cloud_df.write.mode('append').format('hive').saveAsTable('curated_layer_db.word_cloud')
                                else:
                                        word_cloud_df = spark.sparkContext.parallelize(word_list).toDF("string")
                                        word_cloud_df.write.mode('overwrite').format('hive').saveAsTable('curated_layer_db.word_cloud')

                                #### check tweet rate dataframe exists ####
                                if spark._jsparkSession.catalog().tableExists('curated_layer_db', 'word_cloud'):
                                        tweet_rate_df.write.mode('append').format('hive').saveAsTable('curated_layer_db.tweet_rate')
                                else:
                                        tweet_rate_df.write.mode('overwrite').format('hive').saveAsTable('curated_layer_db.tweet_rate')



        # check to see if messages have not been sent in a long time if not then break while loop
        #if time.time() - time_of_msg > 300: # 3 minutes
        #       x = 1

