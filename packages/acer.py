import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.encoder_clf import Encoder
# from stanfordcorenlp import StanfordCoreNLP


class ACER:
    def __init__(self, args, word_dict, inv_dict, device):
        self.args = args
        self.word_dict = word_dict
        self.inv_dict = inv_dict
        self.device = device

        self.con_out_list = []
        self.con_out = np.array([])

        self.loss_class = nn.BCEWithLogitsLoss()  # 多标签的loss
        self.loss_regression = nn.MSELoss()  # 回归的loss
        self.load_ckpt_file = 'checkpoint/param_pmrloss.pkl'
        if args.psy_clf:
            path = 'stanford_nlp/stanford-corenlp-full-2018-10-05'
            # self.nlp = StanfordCoreNLP(path, lang='en')
        self.model = self.load_model()

    def parse(self, sentence):
        sub = []
        res = self.nlp.dependency_parse(sentence)
        print('语法分析:', res)
        for i in range(len(res)):
            if res[i][0] == 'nsubj':
                sub.append(res[i][2])
                # break
        tmp = self.nlp.word_tokenize(sentence)
        re = []
        try:
            for i in range(len(sub)):
                re.append(tmp[sub[i] - 1])
        except Exception as e:
            re = ['none']
        return re

    def label_smoothing(self, inputs, epsilon=0.1):
        ls_target = []
        # if inputs.size()[0] == 1:
        #     inputs = [inputs]
        for i in inputs:
            K = len(i)  # number of channels
            i_ls_target = ((1 - epsilon) * i) + (epsilon / K)
            i_ls_target = i_ls_target.tolist()
            ls_target.append(i_ls_target)
        ls_target = torch.tensor(ls_target)
        return ls_target

    def load_model(self):
        vocab_size = 400000
        dropout = 0.2
        hidden_size = 100
        embedding_dim = 50
        out_size_p = 8
        out_size_m = 5
        out_size_r = 19
        # clip = 5.0
        out_size = {'p': out_size_p, 'm': out_size_m, 'r': out_size_r}

        model = Encoder(vocab_size=vocab_size,
                        embedding_dim=embedding_dim,
                        hidden_size=hidden_size,
                        out_size=out_size,
                        dropout=dropout)
        model = model.to(self.device)

        load_ckpt_file = self.load_ckpt_file
        model.load_state_dict(torch.load(load_ckpt_file))
        model.eval()

        return model

    def calcualate_acer(self, translate_idx, sentence_x, context_x, char_idx, p_score, m_score, r_score):
        '''
        translate_idx(translation_idx):  (1, l)
        '''

        # [[]]  (1, l)
        # 测试时batch_size是1
        batch_size = 1
        loss_pmr = 0
        acer_score = 0
        for i in range(batch_size):
            # for extracted char, transformed to text from idx
            con_out_list = context_x[i].tolist() + sentence_x[i].tolist()   # sentence_input idx & context input idx
            # con_out_list1 = con_out_list + translate_idx[i].tolist()
            # con_out = " ".join([self.inv_dict[w] for w in con_out_list]).replace('bos', '').replace('eos', '').replace(
            #     'pad', '')
            # using Stanford to extract person
            # print('con_out_list: ', con_out_list)
            # print('con_out_list1: ', con_out_list1)
            # print('context: ', con_out)
            # char_extr = self.parse(con_out)[:6]  # char text
            # print('extracted character: ', char_extr)
            # exit()
            char_id = [self.inv_dict[w] for w in char_idx if w != self.word_dict['none']]
            print(char_id)
            # char_id = [self.word_dict[char] for char in char_extr if char in self.word_dict]  # char idx
            with torch.no_grad():
                # 改为一句一句放进去，因为每一句根据抽取的不同人数要复制多次，所以喂入clf的数据要比一个batch多
                for j in range(len(char_id)):
                    sen_pre = translate_idx[i].long().unsqueeze(0)
                    # if type(translate_idx[i]) != list:
                    #     sen_pre_len = [translate_idx[i]]
                    # else:
                    sen_pre_len = torch.tensor([len(translate_idx[i])]).to(self.device).long()
                    con_pre = torch.tensor(con_out_list).to(self.device).long().unsqueeze(0)
                    con_pre_len = torch.tensor([len(con_out_list)]).to(self.device).long()
                    char_pre = torch.tensor([char_idx[j]]).to(self.device).long().unsqueeze(0)  # use NER
                    plutchik_y = torch.tensor(p_score[j]).to(self.device).float().unsqueeze(0)
                    maslow_y = torch.tensor(m_score[j]).to(self.device).float().unsqueeze(0)
                    reiss_y = torch.tensor(r_score[j]).to(self.device).float().unsqueeze(0)

                    p, m, r = self.model(sen_pre, sen_pre_len, con_pre, con_pre_len, char_pre)
                    # p:[1, 8],  m:[1, 5]
                    # print(m)
                    m = torch.sigmoid(m)
                    r = torch.sigmoid(r)
                    # print(m)

                    # 找pmr_y不等于0对应的idx，然后于pmr_pre相减得到差
                    # pmr_y = torch.cat((plutchik_y, maslow_y, reiss_y), 2)[pmr_y != 0]  #
                    pmr_y = torch.cat((plutchik_y, maslow_y, reiss_y), 1)
                    pmr_y_mask = torch.ne(pmr_y, 0)  # pmr_y不等于0的mask
                    pmr_pre = torch.cat((p, m, r), 1)  #
                    pmr_pre = pmr_pre * pmr_y_mask  # 将不比较的元素位置全部变为0
                    # print(pmr_y)
                    # print(pmr_pre)
                    # print(sum(pmr_y_mask).tolist())
                    if sum(sum(pmr_y_mask).tolist()) == 0:
                        break
                    acer_score = sum((pmr_y.squeeze(0) - pmr_pre.squeeze(0)).tolist()) / sum(sum(pmr_y_mask).tolist())
                    # 将预测出来对pmr和标签pmr做对比
        return acer_score
