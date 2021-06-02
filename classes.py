class Article:
    def __init__(self, article):
        self.uid = article['uid']
        self.dct = article['dct']
        self.dates = article['dates']
        self.text = article['text']


class Cluster:
    def __init__(self):
        self.date = None
        self.date_count = 0
        self.articles = []
        self.centroid = None
