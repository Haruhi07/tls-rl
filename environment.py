import env_utils


class Environment:
    def __init__(self, args, keywords):
        self.keywords = set(keywords)

    def count_keyword(self, text):
        word_list = text.lower().split()
        #ret = 0
        #for word in word_list:
        #    if word in self.keywords:
        #        ret += 1
        #return ret
        return len(set(word_list) & self.keywords)
    
    def calc_reward(self):
        text = env_utils.concatenate(self.timelines)
        return self.count_keyword(text)

