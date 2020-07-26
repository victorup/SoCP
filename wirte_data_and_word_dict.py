# utf-8
import nltk
import csv
import numpy as np
import os
import re
import json
from tqdm import tqdm
from packages.vocabulary import build_dict
nltk.download('punkt')
import random
random.seed()


# ##########加载#######
PAD_IDX = 0
UNK_IDX = 1
char_num = 6
p_size = 8
m_size = 5
r_size = 19
PLUTCHIK = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
REISS = ['status', 'approval', 'tranquility', 'competition', 'health', 'family',
         'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity',
         'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']  # 19个reiss
MASLOW = ['physiological', 'stability', 'love', 'esteem', 'spiritual growth']


def search_vector(s):
    z = {}
    for i in eval(s):
        z.update(i)
    return z


def normalization(data):
    _range = np.max(data) - np.min(data)
    if _range != 0:
        return list((data - np.min(data)) / _range)
    else:
        return data


def add_pmr_char(line, char):
    pmr_char_list = []
    pmr_char_list.extend(normalization(search_vector(line[-3])[char]))
    pmr_char_list.extend(search_vector(line[-2])[char])
    pmr_char_list.extend(search_vector(line[-1])[char])
    return pmr_char_list


def add_pmr(line, char):
    plutchik_list = []
    maslow_list = []
    reiss_list = []
    plutchik_list.extend(normalization(search_vector(line[-3])[char]))
    maslow_list.extend(search_vector(line[-2])[char])
    reiss_list.extend(search_vector(line[-1])[char])
    return plutchik_list, maslow_list, reiss_list


def clean_text(text):
    """
    Clean text
    :param text: the string of text
    :return: text string after cleaning
    """
    # acronym
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)

    # spelling correction
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", " facebook ", text)
    text = re.sub(r"facebooks", " facebook ", text)
    text = re.sub(r"facebooking", " facebook ", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r" 1 ", " one ", text)
    text = re.sub(r" 2 ", " two ", text)
    text = re.sub(r" 3 ", " three ", text)
    text = re.sub(r" 4 ", " four ", text)
    text = re.sub(r" 5 ", " five ", text)
    text = re.sub(r" 6 ", " six ", text)
    text = re.sub(r" 7 ", " seven ", text)
    text = re.sub(r" 8 ", " eight ", text)
    text = re.sub(r" 9 ", " nine ", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)

    # punctuation
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)

    # symbol replacement
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)

    # remove extra space
    text = ' '.join(text.split())

    return text


def prepare_data(seq, word_dict):
    words = []
    for word in seq:
        if word in word_dict:
            value = word_dict[word]
        else:
            value = word_dict['<unk>']
        words.append(value)
    return words, len(words)


def load_pmr_data(data_file):
    sentence_all = []
    sentence_x = []
    context_x = []
    sentence_y = []
    character = []
    plutchik_x = []
    maslow_x = []
    reiss_x = []
    char_list_x = []

    with open(data_file, newline='') as csvfile:
        data = csv.reader(csvfile)
        # 遍历每一行
        for idx, line in tqdm(enumerate(data)):
            # title略去
            if idx == 0:
                continue
            # if idx < 99999:
            # if idx < 51:
            # 将char列表控制在char_num个人，多余的人用None来表示
            char_list = eval(line[3])
            if len(char_list) < char_num:  # 小于char_num添加None
                char_list.extend(['none'] * (char_num - len(char_list)))
            elif len(char_list) > char_num:  # 大于char_num只取前char_num个
                char_list = char_list[:char_num]
            char_l = []
            for char in char_list:  # 处理 I (myself)
                if char == 'I (myself)':
                    char = 'I'
                char_l.append(char.lower())  # 一句话char_num个char的列表
            character.append(char_l)

            # sent1-4
            if line[0][-1] != '5':
                # sentence、context输入
                sentence_x.append(["bos"] + nltk.word_tokenize(clean_text(line[1].lower())) + ["eos"])
                context_x.append(["bos"] + nltk.word_tokenize(clean_text(line[2].replace('|', ' ').lower())) + ["eos"])
            # sent2-5
            if line[0][-1] != '1':
                # sentence输出
                sentence_y.append(["bos"] + nltk.word_tokenize(clean_text(line[1].lower())) + ["eos"])
                # # next PMR score
                plutchik = []  # 存储一句话中的6个人物
                maslow = []
                reiss = []
                for i in range(len(char_list)):
                    if char_list[i] != 'none':
                        plutchik_vector, maslow_vector, reiss_vector = add_pmr(line, char_list[i])  # 返回每个人物的pmr分数
                        plutchik.append(plutchik_vector)
                        maslow.append(maslow_vector)
                        reiss.append(reiss_vector)
                    else:
                        plutchik.append([0] * p_size)  # 人物为None，pmr全是0
                        maslow.append([0] * m_size)  # 人物为None，pmr全是0
                        reiss.append([0] * r_size)  # 人物为None，pmr全是0
                plutchik_x.append(plutchik)
                maslow_x.append(maslow)
                reiss_x.append(reiss)
                # char_list_x.append(char_list)
                char_list_x.append(char_l)

            sentence_all.append(["bos"] + nltk.word_tokenize(clean_text(line[1].lower())) + ["eos"])
            build_list = sentence_all + character

    # pmr_char_pre_x: 1个大列表，套19130个句子列表，每个句子列表里有6个人物列表，每个人物列表是pmr数量(8+5+19)
    return build_list, sentence_all, sentence_x, context_x, char_list_x, plutchik_x, maslow_x, reiss_x, sentence_y


def write_word_dict(args, word_dict):
    with open(args.word_dict, 'a') as f:
        json_str = json.dumps(word_dict)
        f.write(json_str)


def write_data(args):
    print('reading data')
    build_list, sentence_all, sentence_x, context_x, char_list_x, plutchik_x, maslow_x, reiss_x, sentence_y \
        = load_pmr_data(args.raw_data_path)
    word_dict, total_words = build_dict(build_list)
    if not os.path.exists(args.word_dict):
        write_word_dict(args, word_dict)

    if not os.path.exists(args.data_path):
        # word2idx
        sx, sxl, cx, cl, sy, syl, ci, cil = [], [], [], [], [], [], [], []
        for i in range(len(sentence_x)):
            sent_x, sent_x_len = prepare_data(sentence_x[i], word_dict)
            con_x, con_x_len = prepare_data(context_x[i], word_dict)
            sent_y, sent_y_len = prepare_data(sentence_y[i], word_dict)
            ch_idx, ch_len = prepare_data(char_list_x[i], word_dict)
            sx.append(sent_x)
            sxl.append(sent_x_len)
            cx.append(con_x)
            cl.append(con_x_len)
            ci.append(ch_idx)
            sy.append(sent_y)
            syl.append(sent_y_len)

        write_path = args.data_path
        with open(write_path, 'w', newline='') as csvfile:
            data = csv.writer(csvfile)
            for i in range(len(sentence_x)):
                data.writerow([sx[i], sxl[i], cx[i], cl[i], ci[i], plutchik_x[i], maslow_x[i], reiss_x[i],
                               sy[i], syl[i]])
