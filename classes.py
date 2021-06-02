class Article:
    def __init__(self, article):
        self.uid = article['uid']
        self.dct = article['dct']
        self.dates = article['dates']
        self.text = article['text']


class Cluster:
    def __init__(self):
        self.date = ""
        self.articles = []
