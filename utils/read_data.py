import pprint
from utils.db_connection import MongodbConnector


class ReadData():

    def twitterDataFiltered(db, filter_key, filet_value):
        twitter_collection = MongodbConnector.connect_collection(db, 'twitter')
        twitterDict_list = twitter_collection.find({filter_key:filet_value})
        dict_list = []
        for twitterDict in twitterDict_list:
            dict = {}
            pprint.pprint(twitterDict)
            dict['source'] = "twitter"
            dict['source_id'] = twitterDict['_id']
            dict['text'] = twitterDict['text']
            dict_list.append(dict)
        return dict_list

    def twitterDataAll(db):
        twitter_collection = MongodbConnector.connect_collection(db, 'twitter')
        twitterDict_list = twitter_collection.find()
        dict_list = []
        for twitterDict in twitterDict_list:
            dict = {}
            #pprint.pprint(twitterDict)
            dict['source'] = "twitter"
            dict['source_id'] = twitterDict['_id']
            dict['text'] = twitterDict['text']
            dict_list.append(dict)
        return dict_list

    def facebookData(self, db, filter_key, filter_value):
        facebook_collection = db.facebook
        facebookDict_list = facebook_collection.find({filter_key:filter_value})
        facebook_list = []
        for facebookDict in facebookDict_list:
            facebook_list.append(facebookDict['text'])
        return facebook_list