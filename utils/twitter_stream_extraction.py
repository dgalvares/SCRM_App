from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
from utils.db_connection import MongodbConnector
from scrm.models import Extracao
import json


db = MongodbConnector.connect_db('socialNet')
collection = MongodbConnector.connect_collection(db, 'twitter')

consumer_key = "xeeRZm1XLRPAkTyMue14hhCtG"
consumer_secret = "AtaxqVV1JenQOxHBQg8yPV8LTWPm5IX2QCz1R47LEEjLryNVH1"
access_token = "2245844254-5CKGOTbuS5js7kpnYXVA70GfYxoznXUp0M9aJcz"
access_token_secret = "apwDkhxoWZAw9g23AJmnOowsvAOHuTwgox0XWZt2XZSYh"


def extrair(filter, max_twitts, extracao_id, extracao):
    twitter_list = []
    running = 0
    class StdOutListener(StreamListener):
        def on_data(self, data):
            twitt = json.loads(data)
            twitt['extracao'] = extracao_id
            # twitt['tempo'] = tempo
            twitter_list.append(collection.insert_one(twitt))
            if collection.find({"extracao": extracao_id}).count() > int(max_twitts) - 1:
                extracao.concluido = True
                extracao.quantidade = collection.find({"extracao": extracao_id}).count()
                extracao.save()
                print("fim")
                return False
            else:
                print("rodando")
                return True

        def on_error(selfs, status):
            print(status)

    stream_listener = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, stream_listener)
    stream.filter(track=[""+filter], async=False)

