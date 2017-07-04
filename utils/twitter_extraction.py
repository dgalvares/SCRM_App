from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from utils.db_connection import MongodbConnector
import json

db = MongodbConnector.connect_db('socialNet')
collection = MongodbConnector.connect_collection(db,'twitter')

consumer_key="xeeRZm1XLRPAkTyMue14hhCtG"
consumer_secret="AtaxqVV1JenQOxHBQg8yPV8LTWPm5IX2QCz1R47LEEjLryNVH1"
access_token="2245844254-5CKGOTbuS5js7kpnYXVA70GfYxoznXUp0M9aJcz"
access_token_secret="apwDkhxoWZAw9g23AJmnOowsvAOHuTwgox0XWZt2XZSYh"

class StdOutListener(StreamListener):

    def on_data(self, data):
        twitt = collection.insert_one(json.loads(data))
        print(twitt.inserted_id)
        return(True)

    def on_error(self,status):
        print(status)

if __name__ == '__main__':
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    stream.filter(track=['Londres'])
