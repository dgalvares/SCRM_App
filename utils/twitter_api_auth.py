import tweepy

consumer_key = "xeeRZm1XLRPAkTyMue14hhCtG"
consumer_secret = "AtaxqVV1JenQOxHBQg8yPV8LTWPm5IX2QCz1R47LEEjLryNVH1"
access_token = "2245844254-5CKGOTbuS5js7kpnYXVA70GfYxoznXUp0M9aJcz"
access_token_secret = "apwDkhxoWZAw9g23AJmnOowsvAOHuTwgox0XWZt2XZSYh"

def twitter_auth():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api