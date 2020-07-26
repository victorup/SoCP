from torch.utils.data import Dataset
import csv
import numpy as np


class MyDataset_pmr(Dataset):
    def __init__(self, data_path):
        self.sentence_x = []
        self.sentence_x_len = []
        self.context_x = []
        self.context_x_len = []
        self.char_idx_next = []
        self.plutchik_x = []
        self.maslow_x = []
        self.reiss_x = []
        self.sentence_y = []
        self.sentence_y_len = []
        with open(data_path, "r", encoding="utf8") as csvfile:
            self.data = csv.reader(csvfile)
            for idx, line in enumerate(self.data):
                self.sentence_x.append(eval(line[0]))
                self.sentence_x_len.append(eval(line[1]))
                self.context_x.append(eval(line[2]))
                self.context_x_len.append(eval(line[3]))
                self.char_idx_next.append(eval(line[4]))
                self.plutchik_x.append(eval(line[5]))
                self.maslow_x.append(eval(line[6]))
                self.reiss_x.append(eval(line[7]))
                self.sentence_y.append(eval(line[8]))
                self.sentence_y_len.append(eval(line[9]))

    def __len__(self):
        return len(self.sentence_x)

    def __getitem__(self, idx):
        return self.sentence_x[idx], self.sentence_x_len[idx], self.context_x[idx], self.context_x_len[idx], \
               self.char_idx_next[idx], self.plutchik_x[idx], self.maslow_x[idx], self.reiss_x[idx], \
               self.sentence_y[idx], self.sentence_y_len[idx]


class MyDataset_clf(Dataset):
    def __init__(self, data_path):
        self.sentence_x = []
        self.sentence_x_len = []
        self.plutchik = []
        with open(data_path, "r", encoding="utf8") as csvfile:
            data = csv.reader(csvfile)
            for idx, line in enumerate(data):
                char = [i for i in eval(line[4]) if i != 2]
                p = []
                for i in eval(line[4]):  # 遍历人物列表，人物列表中存在的可能有的只有M或R没有P，所以存在plutchik是空情感
                    if i != 2:
                        self.sentence_x.append([i] + eval(line[8]))
                        self.sentence_x_len.append(eval(line[9]) + 1)
                        p.append(0)
                idx = 0
                for char in eval(line[5]):  # 遍历6个人物的plutchik
                    if sum(char) != 0:  # 如果这个人物有Plutchik
                        p[idx] = char.index(
                            max(char)) + 1  # 找到是1分对应的索引+1赋值给p，因为索引0我们想对应着空情感 标签[0~9]，0是无，1-9分别对应plutchik
                        idx += 1
                self.plutchik.extend(p)

    def __len__(self):
        return len(self.sentence_x)

    def __getitem__(self, idx):
        return self.sentence_x[idx], self.sentence_x_len[idx], self.plutchik[idx]


def get_pmr_np(args, pmr_all, word_dict):
    if args.gpt2:
        pmr_np = np.zeros((len(pmr_all), 1)).astype('int32')
        tokenizer = word_dict
        for idx, pmr in enumerate(pmr_all):
            value = tokenizer.encode(pmr)[:1]
            pmr_np[idx, :] = np.array(value)
    else:
        pmr_np = np.zeros((len(pmr_all), 1)).astype('int32')
        for idx, pmr in enumerate(pmr_all):
            if pmr in word_dict:
                value = word_dict[pmr]
            else:
                value = word_dict['unk']
            pmr_np[idx, :] = np.array(value)
    return pmr_np


def get_pmr(args, word_dict):
    plutchik = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    maslow = ['physiological', 'love', 'spiritual', 'esteem', 'stability']
    reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family',
             'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity',
             'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']  # 19个reiss
    p_np = get_pmr_np(args, plutchik, word_dict)
    m_np = get_pmr_np(args, maslow, word_dict)
    r_np = get_pmr_np(args, reiss, word_dict)

    return p_np, m_np, r_np
