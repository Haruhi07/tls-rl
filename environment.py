import env_utils


class Environment:
    def __init__(self, tokenizer, clusters, keywords, t_length):
        self.tokenizer = tokenizer
        self.clusters = clusters
        self.keywords = set(keywords)
        self.t_length = t_length
        self.timelines = [{"date": None, "text": ""} for i in range(t_length)]
        self.date_pt = 0

    def observation(self):
        return self.clusters[self.date_pt], self.timelines[self.date_pt]

    def count_keyword(self, text):
        word_list = text.lower().split()
        ret = 0
        for word in word_list:
            if word in self.keywords:
                ret += 1
        return ret
        #return len(set(word_list) & self.keywords)

    def reset(self):
        self.timelines = [{"date": None, "text": ""} for i in range(self.t_length)]
        self.date_pt = 0
    
    def calc_reward(self):
        text = env_utils.concatenate(self.timelines)
        return self.count_keyword(text)
        
    
    def step(self, action):
        new_word = self.tokenizer.decode(action, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print("new word = ", new_word)
        done = False
        if action == 1:
            self.timelines[self.date_pt]["date"] = self.clusters[self.date_pt].date
            self.date_pt += 1
            if self.date_pt >= len(self.timelines):
                done = True
        elif self.timelines[self.date_pt]["text"] == '':
            self.timelines[self.date_pt]["text"] = new_word
        else:
            self.timelines[self.date_pt]["text"] += " " + new_word
        reward = self.calc_reward()
        state = (self.clusters[self.date_pt], self.timelines[self.date_pt])
        return state, reward, done
        
