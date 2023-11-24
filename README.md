# Sentiment-Analysis

We're using Reddit to collect data. First you need to create developer account on reddit and get your secret and client id to run the code. The url to create reddit app is https://www.reddit.com/prefs/apps. If the url is not directly accacible, please login to reddit account and then manually inter the url. It will ask you to create an application, which you need to do and you will get your secrets after. 

## Data collection
you will need reddit account as explained above, you can try different subreddits and number of posts you want to fetch.It will get saved in as json file in data folder with text_based_post.txt file name. 

## labelling
This code uses pre-trained model to label collected data and saves in data folder with labelled_data.json file name

## preprocessing and modeling
Here, we pre process the data and save is as tokenised data in another json file in data folder named updated_json_file.json. This file then gets converted to dataframe and used to train model. 
