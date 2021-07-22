import env_utils


class Environment:
    def __init__(self, tokenizer, clusters, keywords, t_length):
        print(keywords)
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
        return len(set(word_list) & self.keywords)

    def reset(self):
        self.timelines = [{"date": None, "text": ""} for i in range(self.t_length)]
        self.date_pt = 0
    
    def calc_reward(self):
        text = env_utils.concatenate(self.timelines)
        return self.count_keyword(text)
        
    
    def step(self, action):
        new_word = self.tokenizer.decode([action], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        done = False
        if action == 1:
            self.date_pt += 1
            if self.date_pt >= len(self.timelines):
                done = True
        else:
            self.timelines[self.date_pt]["text"] += " " + new_word
        reward = self.calc_reward()
        state = (self.clusters[self.date_pt], self.timelines[self.date_pt])
        return state, reward, done
        
