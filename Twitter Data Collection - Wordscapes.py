#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[15]:


import tweepy
from textblob import TextBlob
import emoji
import re
import string
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


# # Data Collection

# In[2]:


ACCESS_TOKEN = "2378032825-QNQ4zETCM2hcIuP80deSHGdTXcrzE7vysvkdcNz"
ACCESS_TOKEN_SECRET = "JIoaBDifICUapIXWGE1x636itkIrHHxs7he0WwiClY6kv"
CONSUMER_KEY = "xU5CZUCtq2jNiopmogUvegudT"
CONSUMER_SECRET = "goL8kGtG5MMe5oKjivh3MT6Ic88iajrjqJ26NpJj5upiJFB3Pa"


# In[3]:


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)


# In[4]:


api = tweepy.API(auth)


# In[5]:


query = 'wordscapes' # set 'wordscapes' as a search query
max_tweets = 1000


# In[6]:


searched_tweets = [status for status in tweepy.Cursor(api.search, q=query, tweet_mode="extended").items(max_tweets)]


# Helpful resource: [structure-of-tweepy-status-object.json](https://gist.github.com/dev-techmoe/ef676cdd03ac47ac503e856282077bf2)

# # Data Parsing

# In[10]:


def get_tweet_type(tweet):
    """ Get tweet type: tweet, retweet, quote, reply
    Details for tweet types:  https://gwu-libraries.github.io/sfm-ui/posts/2016-11-10-twitter-interaction
    """
    if hasattr(tweet, 'retweeted_status'):
        return "retweet"
    elif hasattr(tweet, 'quoted_status'):
        return "quote"
    elif tweet.in_reply_to_status_id_str is not None:
        return "reply"
    else:
        return "tweet"


# In[11]:


def replace_urls(in_string, replacement=None):
    """Replace URLs in strings.

    Args:
        in_string (str): string to filter
        replacement (str or None): replacment text. defaults to ''

    Returns:
        str
    """
    replacement = '' 
    pattern = re.compile('(https?://)?(\w*[.]\w+)+([/?=&]+\w+)*')
    return re.sub(pattern, replacement, in_string)


# In[12]:


def remove_emoji(in_string):
    return emoji.get_emoji_regexp().sub(u' ', in_string)


# In[13]:


def get_original_tweet_id(tweet):
    """ Return original tweet id if tweet type is retweet. Otherwise, return its own id.
    """
    if get_tweet_type(tweet) == "retweet":
        return tweet.retweeted_status.id
    else:
        return tweet.id


# In[16]:


data_df = pd.DataFrame([{"date": x.created_at,
                         "text": x.full_text,
                         "clean_text": remove_emoji(replace_urls(x.full_text)),
                         "tweet_id": x.id,
                         "if_retweet": get_original_tweet_id(x),
                         "user": x.user.screen_name,
                         "user_bio": x.user.description,
                         "user_id": x.user.id,
                         "at_mentions": [u["screen_name"] for u in x.entities['user_mentions']],
                         "hashtags": [u["text"] for u in x.entities['hashtags']],
                         "language": x.lang,
                         "type": get_tweet_type(x),
                         "retweet_count": x.retweet_count,
                         "favorite_count": x.favorite_count,
                         "polarity": TextBlob(x.full_text).sentiment.polarity} for x in searched_tweets]).set_index("date")


# In[98]:


print(data_df.iloc[1])


# In[32]:


data_df[["language","tweet_id"]].groupby("language").count()


# In[34]:


data_df[["type","tweet_id"]].groupby("type").count()


# In[161]:


def get_emotion(df):
    """
    polarity = 0: neutral;
    polarity > 0: positive;
    polarity < 0: negative.
    """
    if df['polarity'] == 0:
        return 'neutral'
    elif df['polarity'] > 0.0:
        return 'positive'
    else:
        return 'negative'


# In[162]:


data_df['emotion'] = data_df.apply(get_emotion, axis=1)


# In[178]:


data_df.info()


# In[163]:


data_df.head()


# In[19]:


retweet_list = []


# In[20]:


for tweet in searched_tweets:
    if get_tweet_type(tweet) == "retweet":
        retweet_list.append(tweet)


# In[166]:


retweet_df = pd.DataFrame([{"date": x.retweeted_status.created_at,
                         "text": x.retweeted_status.full_text,
                         "clean_text": remove_emoji(replace_urls(x.retweeted_status.full_text)),
                         "tweet_id": x.retweeted_status.id,
                         "user": x.retweeted_status.user.screen_name,
                         "user_bio": x.retweeted_status.user.description,
                         "user_id": x.retweeted_status.user.id,
                         "at_mentions": [u["screen_name"] for u in x.retweeted_status.entities['user_mentions']],
                         "hashtags": [u["text"] for u in x.retweeted_status.entities['hashtags']],
                         "language": x.retweeted_status.lang,
                         "type": get_tweet_type(x),
                         "retweet_count": x.retweeted_status.retweet_count,
                         "favorite_count": x.retweeted_status.favorite_count,
                         "polarity": TextBlob(x.retweeted_status.full_text).sentiment.polarity} for x in retweet_list]).set_index("date")


# In[167]:


retweet_df['emotion'] = retweet_df.apply(get_emotion, axis=1)


# In[168]:


retweet_df.head()


# In[170]:


retweet_df = retweet_df.drop_duplicates(subset='tweet_id', keep="last")


# In[177]:


retweet_df.info()


# # Data Normalization

# In[38]:


df_for_norm = data_df[data_df['language'] == 'en']


# In[40]:


clean_text = df_for_norm['clean_text'].tolist()
mentions = df_for_norm['at_mentions'].tolist()
hashtags = df_for_norm['hashtags'].tolist()
tweet_id = df_for_norm['tweet_id'].tolist()


# In[152]:


def remove_mh(in_string, mentions, hashtags):
    """Remove at_mentions and hashtags from tweets.
    """
    mentions = ['@'+s for s in mentions]
    hashtags = ['#'+s for s in hashtags]
    stopwords = mentions+hashtags
    querywords = in_string.split()
    resultwords  = [word for word in querywords if word not in stopwords]
    result = ' '.join(resultwords)
    return result


# In[153]:


def normalize(in_string, mentions, hashtags):
    """Normalize tweets' text.
    """
    in_string = remove_mh(in_string, mentions, hashtags)
    result = in_string.lower() # To Lower
    result = result.translate(str.maketrans(' ',' ',string.punctuation)) # Remove Puntuation
    result = result.strip() # Remove White Space
    lemmatizer=WordNetLemmatizer() # Lemmatization
    stop_words = set(stopwords.words('english')) # Remove Stop Words, at_mentions and hashtags
    tokens = word_tokenize(result)
    result = [lemmatizer.lemmatize(i) for i in tokens if not i in stop_words]
    return result


# In[154]:


list_norm = []


# In[155]:


for i in range(len(clean_text)):
    dic_norm = {
        "tweet_id": tweet_id[i],
        "tokens": normalize(clean_text[i], mentions[i], hashtags[i])
    }
    list_norm.append(dic_norm)


# In[156]:


df_norm = pd.DataFrame(list_norm)


# In[157]:


df_norm.head()


# In[158]:


df_norm_clean = df_norm.tokens.apply(pd.Series)  .merge(df_norm, right_index = True, left_index = True)  .drop(["tokens"], axis = 1)  .melt(id_vars = ['tweet_id'], value_name = "tokens")  .drop("variable", axis = 1)  .dropna()


# In[160]:


df_norm_clean.head()


# In[173]:


df_freq = df_norm_clean[["tokens","tweet_id"]].groupby("tokens").count()


# In[192]:


df_freq.head()


# # Data Exportation

# In[180]:


tweet_data = data_df.drop(['if_retweet'], axis=1)


# In[186]:


tweet_data = tweet_data[tweet_data['type'] != 'retweet']


# In[187]:


tweet_data.info()


# In[188]:


frames = [tweet_data, retweet_df]


# In[189]:


df_comb = pd.concat(frames)


# In[190]:


df_comb.info()


# In[193]:


with pd.ExcelWriter('output.xlsx') as writer: 
    retweet_df.to_excel(writer, sheet_name='Retweet')
    data_df.to_excel(writer, sheet_name='Raw Data')
    df_comb.to_excel(writer, sheet_name='Combination')
    df_freq.to_excel(writer, sheet_name='Frequency')
    df_norm_clean.to_excel(writer, sheet_name='Tokens')


# In[ ]:




