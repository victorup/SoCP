import torch
import argparse
import torch.nn as nn
import json
import sys
from configparser import ConfigParser
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from packages.function import create_logger
from dataloder import MyDataset_pmr, get_pmr  # 可通用
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from model.attention import Attention
from model.pmr_clf import PMRClf
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge import Rouge

from model.encoder import Encoder
from model.decoder import Decoder  # 可通用
from model.story_seq2seq import PreStory  # 可通用

plutchik = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
reiss = ['status', 'approval', 'tranquility', 'competition', 'health', 'family',
         'romance', 'food', 'indep', 'power', 'order', 'curiosity', 'serenity',
         'honor', 'belonging', 'contact', 'savings', 'idealism', 'rest']  # 19个reiss
maslow = ['physiological', 'stability', 'love', 'esteem', 'spiritual growth']
pre_list = []
y_list = []


def elapsed_time(pre_time):
    import time
    elapsed = time.time()
    print("spend time: {:.2f}s".format(elapsed - pre_time))
    return elapsed


def display_attention(sentence, translation, attention):
    fig = plt.figure(figsize=(15, 15))  # 图片大小
    ax = fig.add_subplot(111)

    attention = attention.squeeze(0).cpu().detach().numpy()[:len(translation)]
    # attention = attention.cpu().detach().numpy()

    cax = ax.matshow(attention, cmap='bone')

    ax.tick_params(labelsize=15)  # 标签大小
    # ax.set_xticklabels([''] + ['<sos>'] + [t.lower() for t in sentence] + ['<eos>'], rotation=45)
    ax.set_xticklabels([''] + [t.lower() for t in sentence], rotation=90)  # label的角度
    ax.set_yticklabels([''] + translation)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def translate_story_dev(idx, i, args, word_dict, model, p_np, m_np, r_np, x, con, y, char_idx, inv_dict, p_score,
                        m_score,
                        r_score, device):
    if args.self_test:
        if idx == 0:
            x_sent = x
            con_sent = con
            y_sent = y
            x = [[word_dict[w] for w in x.split()]]
            con = [[word_dict[w] for w in con.split()]]
            mini_x = torch.tensor(x[i]).long().to(device).unsqueeze(0)  # (b, len)
            mini_x_len = torch.tensor(len(x[i])).long().to(device).unsqueeze(0)  # (b)
            mini_con = torch.tensor(con[i]).long().to(device).unsqueeze(0)  # (b, len)
            mini_con_len = torch.tensor(len(con[i])).long().to(device).unsqueeze(0)  # (b)
        else:
            x = [[3] + x[0] + [4]]
            x_sent = " ".join([inv_dict[w] for w in x[0]]).replace('bos', '').replace('eos', '').strip()
            con_sent = " ".join([inv_dict[w] for w in con[0]]).replace('bos', '').replace('eos', '').strip()
            # sen 
            mini_x = torch.tensor(x[0]).long().to(device).unsqueeze(0)  # (b, len) 
            mini_x_len = torch.tensor(len(x[0])).long().to(device).unsqueeze(0)  # (b) 
            # context
            mini_con = torch.tensor(con[0]).long().to(device).unsqueeze(0)  # (b, len) 
            mini_con_len = torch.tensor(len(con[0])).long().to(device).unsqueeze(0)  # (b) 
            con = [con[0][:-1] + x[0][1:]]  # 将之前的句子作为context
    else:
        if idx == 0:  # 第一次进来的i
            x_sent = " ".join([inv_dict[w] for w in x[i]]).replace('bos', '').replace('eos', '').strip()
            con_sent = " ".join([inv_dict[w] for w in con[i]]).replace('bos', '').replace('eos', '').strip()
            # sen 
            mini_x = torch.tensor(x[i]).long().to(device).unsqueeze(0)  # (b, len) 
            mini_x_len = torch.tensor(len(x[i])).long().to(device).unsqueeze(0)  # (b) 
            # context 
            mini_con = torch.tensor(con[i]).long().to(device).unsqueeze(0)  # (b, len) 
            mini_con_len = torch.tensor(len(con[i])).long().to(device).unsqueeze(0)  # (b) 
            con = [x[i]]  # 将之前的句子作为context  
        else:
            x = [[3] + x[0] + [4]]
            x_sent = " ".join([inv_dict[w] for w in x[0]]).replace('bos', '').replace('eos', '').strip()
            con_sent = " ".join([inv_dict[w] for w in con[0]]).replace('bos', '').replace('eos', '').strip()
            # sen 
            mini_x = torch.tensor(x[0]).long().to(device).unsqueeze(0)  # (b, len) 
            mini_x_len = torch.tensor(len(x[0])).long().to(device).unsqueeze(0)  # (b) 
            # context
            mini_con = torch.tensor(con[0]).long().to(device).unsqueeze(0)  # (b, len) 
            mini_con_len = torch.tensor(len(con[0])).long().to(device).unsqueeze(0)  # (b) 
            con = [con[0][:-1] + x[0][1:]]  # 将之前的句子作为context

    mini_char_idx = torch.tensor(char_idx[i]).long().to(device).unsqueeze(0)  # (b,6)
    mini_p_score = torch.tensor(p_score[i]).float().to(device).unsqueeze(0)
    mini_m_score = torch.tensor(m_score[i]).float().to(device).unsqueeze(0)
    mini_r_score = torch.tensor(r_score[i]).float().to(device).unsqueeze(0)
    mini_p_id = torch.from_numpy(p_np).long().to(device)  # (32, 1)
    mini_m_id = torch.from_numpy(m_np).long().to(device)  # (32, 1)
    mini_r_id = torch.from_numpy(r_np).long().to(device)  # (32, 1)

    # logger.info('x{}: {}'.format(i, x_sent))
    if args.gpu_para and args.use_cuda:
        translation, attn, _, _, _ = model.module.translate(args, mini_x, mini_x_len, mini_con, mini_con_len,
                                                                mini_char_idx,
                                                                mini_p_id, mini_m_id, mini_r_id,
                                                                mini_p_score, mini_m_score, mini_r_score)
    else:
        translation, attn, _, _, _ = model.translate(args, mini_x, mini_x_len, mini_con, mini_con_len, mini_char_idx,
                                                     mini_p_id, mini_m_id, mini_r_id,
                                                     mini_p_score, mini_m_score, mini_r_score)


    translation_word = [inv_dict[i] for i in translation.data.cpu().numpy().reshape(-1)]
    trans = []
    for word in translation_word:
        if word != "eos":
            trans.append(word)
        else:
            break

    # if idx < 10:
    if idx < 0:
        char0 = [[plutchik[j] for j in range(len(p_score[i][0])) if p_score[i][0][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][0])) if m_score[i][0][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][0])) if r_score[i][0][j] != 0]]
        char1 = [[plutchik[j] for j in range(len(p_score[i][1])) if p_score[i][1][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][1])) if m_score[i][1][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][1])) if r_score[i][1][j] != 0]]
        char2 = [[plutchik[j] for j in range(len(p_score[i][2])) if p_score[i][2][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][2])) if m_score[i][2][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][2])) if r_score[i][2][j] != 0]]
        char3 = [[plutchik[j] for j in range(len(p_score[i][3])) if p_score[i][3][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][3])) if m_score[i][3][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][3])) if r_score[i][3][j] != 0]]
        char4 = [[plutchik[j] for j in range(len(p_score[i][4])) if p_score[i][4][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][4])) if m_score[i][4][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][4])) if r_score[i][4][j] != 0]]
        char5 = [[plutchik[j] for j in range(len(p_score[i][5])) if p_score[i][5][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][5])) if m_score[i][5][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][5])) if r_score[i][5][j] != 0]]

        # char0
        p_score0 = [format(p_score[i][0][j], '.2f') for j in range(len(p_score[i][0])) if p_score[i][0][j] != 0]
        weighted_plutchik0 = [i for i in zip(p_score0, char0[0])]
        logger.info('char0: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik0, char0[1], char0[2]))
        # char1
        p_score1 = [format(p_score[i][1][j], '.2f') for j in range(len(p_score[i][1])) if p_score[i][1][j] != 0]
        weighted_plutchik1 = [i for i in zip(p_score1, char1[0])]
        logger.info('char1: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik1, char1[1], char1[2]))
        # char2
        p_score2 = [format(p_score[i][2][j], '.2f') for j in range(len(p_score[i][2])) if p_score[i][2][j] != 0]
        weighted_plutchik2 = [i for i in zip(p_score2, char2[0])]
        logger.info('char2: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik2, char2[1], char2[2]))
        # char3
        p_score3 = [format(p_score[i][3][j], '.2f') for j in range(len(p_score[i][3])) if p_score[i][3][j] != 0]
        weighted_plutchik3 = [i for i in zip(p_score3, char3[0])]
        logger.info('char3: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik3, char3[1], char3[2]))
        # char4
        p_score4 = [format(p_score[i][4][j], '.2f') for j in range(len(p_score[i][4])) if p_score[i][4][j] != 0]
        weighted_plutchik4 = [i for i in zip(p_score4, char4[0])]
        logger.info('char4: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik4, char4[1], char4[2]))
        # char5
        p_score5 = [format(p_score[i][5][j], '.2f') for j in range(len(p_score[i][5])) if p_score[i][5][j] != 0]
        weighted_plutchik5 = [i for i in zip(p_score5, char5[0])]
        logger.info('char5: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik5, char5[1], char5[2]))

    if idx == 0:
        # logger.info('context{}: {}'.format(i, con_sent))
        logger.info('x{}: {}'.format(i, x_sent))
    logger.info('pre_sent{}: {}'.format(i, " ".join(trans)) + '\n')
    return translation.tolist(), con


def translate_dev(i, args, word_dict, model, clf_model, p_np, m_np, r_np, x, con, y, char_idx, inv_dict, p_score, m_score, r_score,
                  device, b1, b2, b3, b4, r1, r2, rl, aser, ref_writer, hypo_writer):
    if args.test_story:
        x_sent = " ".join([inv_dict[w] for w in x[i]]).replace('bos', '').replace('eos', '').strip()
        con_sent = " ".join([inv_dict[w] for w in con[i]]).replace('bos', '').replace('eos', '').strip()
        y_sent = 'none'
    else:
        if args.self_test:
            x_sent = x
            con_sent = con
            y_sent = y
            x = [[word_dict[w] for w in x.split()]]
            con = [[word_dict[w] for w in con.split()]]
        else:
            if args.gpt2:
                x_sent = " ".join([word_dict.decode(w) for w in x[i]]).replace('bos', '').replace('eos', '').strip()
                con_sent = " ".join([word_dict.decode(w) for w in con[i]]).replace('bos', '').replace('eos', '').strip()
                y_sent = " ".join([word_dict.decode(w) for w in y[i]]).replace('bos', '').replace('eos', '').strip()
            else:
                x_sent = " ".join([inv_dict[w] for w in x[i]]).replace('bos', '').replace('eos', '').strip()
                con_sent = " ".join([inv_dict[w] for w in con[i]]).replace('bos', '').replace('eos', '').strip()
                y_sent = " ".join([inv_dict[w] for w in y[i]]).replace('bos', '').replace('eos', '').strip()

    mini_x = torch.tensor(x[i]).long().to(device).unsqueeze(0)  # (b, len)
    mini_x_len = torch.tensor(len(x[i])).long().to(device).unsqueeze(0)  # (b)
    mini_con = torch.tensor(con[i]).long().to(device).unsqueeze(0)  # (b, len)
    mini_con_len = torch.tensor(len(con[i])).long().to(device).unsqueeze(0)  # (b)
    mini_char_idx = torch.tensor(char_idx[i]).long().to(device).unsqueeze(0)  # (b,6)
    mini_p_score = torch.tensor(p_score[i]).float().to(device).unsqueeze(0)[:, :args.char_num]  # (b,6,8)
    mini_m_score = torch.tensor(m_score[i]).float().to(device).unsqueeze(0)[:, :args.char_num]
    mini_r_score = torch.tensor(r_score[i]).float().to(device).unsqueeze(0)[:, :args.char_num]
    mini_p_id = torch.from_numpy(p_np).long().to(device)  # (32, 1)
    mini_m_id = torch.from_numpy(m_np).long().to(device)  # (32, 1)
    mini_r_id = torch.from_numpy(r_np).long().to(device)  # (32, 1)

    # logger.info('x{}: {}'.format(i, x_sent))
    if args.encoder_merge and args.context:
        # [bos + context + sentence + eos]
        mini_x = mini_x[mini_x != word_dict['bos']].view(mini_x.size(0), -1)
        mini_con = mini_con[mini_con != word_dict['eos']].view(mini_con.size(0), -1)
        mini_x = torch.cat([mini_con, mini_x], 1)
    if args.gpu_para and args.use_cuda:
        translation_idx, attn, p_attn, m_attn, r_attn = model.module.translate(args, mini_x, mini_x_len, mini_con,
                                                                                   mini_con_len, mini_char_idx,
                                                                                   mini_p_id, mini_m_id, mini_r_id,
                                                                                   mini_p_score, mini_m_score, mini_r_score)
    else:
        translation_idx, attn, p_attn, m_attn, r_attn = model.translate(args, mini_x, mini_x_len, mini_con,
                                                                            mini_con_len, mini_char_idx,
                                                                            mini_p_id, mini_m_id, mini_r_id,
                                                                            mini_p_score, mini_m_score, mini_r_score)
        # translation_idx: torch.Size([1, 20])

    translation = [inv_dict[i] for i in translation_idx.data.cpu().numpy().reshape(
        -1)]  # <class 'list'>: ["robert's company", 'twin boys', 'yanked', 'singer', 'convinced', 'tables', 'nickname', 'buyer', 'the baby', 'trump', 'trump', 'oldest', 'fooled', 'stanley', 'draft', 'pile', 'television', 'sheepishly', 'clung', 'case']
    # translation: ["company", 'boys', ... '']

    trans = []  # 没有eos的句子  ["company", 'boys', ... '']
    # 出现eos后截断句子
    for word in translation:
        if word != "eos":
            trans.append(word)
        else:
            break
    trans_idx = translation_idx.tolist()[0]
    if 4 not in trans_idx:
        trans_idx.append(4)

    # ##### psychological state clf ######
    '''
    clf_x = []
    clf_x_len = []
    clf_p = []
    idx = 0
    for char in char_idx[i]:
        if char != 2:
            clf_x.append([char] + [3] + trans_idx[:trans_idx.index(4)] + [4])
            clf_x_len.append(len(trans) + 3)
            clf_p.append(0)
    for char in p_score[i]:  # 遍历6个人物的plutchik
        if sum(char) != 0:  # 如果这个人物有Plutchik
            clf_p[idx] = char.index(max(char)) + 1  # 找到是1分对应的索引+1赋值给p，因为索引0我们想对应着空情感 标签[0~9]，0是无，1-9分别对应plutchik
            idx += 1
    for i in range(len(clf_x)):
        mini_clf_x = torch.tensor(clf_x[i]).long().to(device).unsqueeze(0)
        mini_clf_x_len = torch.tensor(len(clf_x[i])).long().to(device).unsqueeze(0)
        mini_clf_p = torch.tensor(clf_p[i]).long().to(device).unsqueeze(0)
        pre_y = clf_model.predict_p(mini_clf_x, mini_clf_x_len, mini_clf_p)
        if pre_y.item() == mini_clf_p.item():
            aser += 1
    # '''
    # sys.exit()
    display_attention(plutchik+maslow+reiss, trans, p_attn)

    # ref_writer.write(y_sent + '\n')
    # hypo_writer.write(' '.join(trans) + '\n')
    # reference = y_sent.split()  # ["a", 'b', ... '']
    # candidate = trans  # ["a", 'b', ... '']
    '''
    # use evaluation respectively
    # '''
    '''
    method = SmoothingFunction().method7
    Bleu_1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0), smoothing_function=method)
    Bleu_2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0), smoothing_function=method)
    Bleu_3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0), smoothing_function=method)
    Bleu_4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=method)
    b1 = b1 + Bleu_1
    b2 = b2 + Bleu_2
    b3 = b3 + Bleu_3
    b4 = b4 + Bleu_4
    
    reference = [y_sent]  # ['i love you']
    candidate = [' '.join(trans)]  # ['i love you']
    rouge = Rouge()
    rouge_score = rouge.get_scores(candidate, reference, )
    rouge_1 = rouge_score[0]["rouge-1"]['r']
    rouge_2 = rouge_score[0]["rouge-2"]['r']
    rouge_l = rouge_score[0]["rouge-l"]['r']
    r1 = r1 + rouge_1
    r2 = r2 + rouge_2
    rl = rl + rouge_l
    '''
    '''
    if i < 10:
        char0 = [[plutchik[j] for j in range(len(p_score[i][0])) if p_score[i][0][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][0])) if m_score[i][0][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][0])) if r_score[i][0][j] != 0]]
        char1 = [[plutchik[j] for j in range(len(p_score[i][1])) if p_score[i][1][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][1])) if m_score[i][1][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][1])) if r_score[i][1][j] != 0]]
        char2 = [[plutchik[j] for j in range(len(p_score[i][2])) if p_score[i][2][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][2])) if m_score[i][2][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][2])) if r_score[i][2][j] != 0]]
        char3 = [[plutchik[j] for j in range(len(p_score[i][3])) if p_score[i][3][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][3])) if m_score[i][3][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][3])) if r_score[i][3][j] != 0]]
        char4 = [[plutchik[j] for j in range(len(p_score[i][4])) if p_score[i][4][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][4])) if m_score[i][4][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][4])) if r_score[i][4][j] != 0]]
        char5 = [[plutchik[j] for j in range(len(p_score[i][5])) if p_score[i][5][j] != 0]] + [
            [maslow[j] for j in range(len(m_score[i][5])) if m_score[i][5][j] != 0]] + [
                    [reiss[j] for j in range(len(r_score[i][5])) if r_score[i][5][j] != 0]]

        # char0
        p_score0 = [format(p_score[i][0][j], '.2f') for j in range(len(p_score[i][0])) if p_score[i][0][j] != 0]
        weighted_plutchik0 = [i for i in zip(p_score0, char0[0])]
        if weighted_plutchik0 != [] and char0[1] != [] and char0[2] != []:
            logger.info('char0: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik0, char0[1], char0[2]))
        # char1
        p_score1 = [format(p_score[i][1][j], '.2f') for j in range(len(p_score[i][1])) if p_score[i][1][j] != 0]
        weighted_plutchik1 = [i for i in zip(p_score1, char1[0])]
        if weighted_plutchik1 != [] and char1[1] != [] and char1[2] != []:
            logger.info('char1: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik1, char1[1], char1[2]))
        # char2
        p_score2 = [format(p_score[i][2][j], '.2f') for j in range(len(p_score[i][2])) if p_score[i][2][j] != 0]
        weighted_plutchik2 = [i for i in zip(p_score2, char2[0])]
        if weighted_plutchik2 != [] and char2[1] != [] and char2[2] != []:
            logger.info('char2: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik2, char2[1], char2[2]))
        # char3
        p_score3 = [format(p_score[i][3][j], '.2f') for j in range(len(p_score[i][3])) if p_score[i][3][j] != 0]
        weighted_plutchik3 = [i for i in zip(p_score3, char3[0])]
        if weighted_plutchik3 != [] and char3[1] != [] and char3[2] != []:
            logger.info('char3: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik3, char3[1], char3[2]))
        # char4
        p_score4 = [format(p_score[i][4][j], '.2f') for j in range(len(p_score[i][4])) if p_score[i][4][j] != 0]
        weighted_plutchik4 = [i for i in zip(p_score4, char4[0])]
        if weighted_plutchik4 != [] and char4[1] != [] and char4[2] != []:
            logger.info('char4: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik4, char4[1], char4[2]))
        # char5
        p_score5 = [format(p_score[i][5][j], '.2f') for j in range(len(p_score[i][5])) if p_score[i][5][j] != 0]
        weighted_plutchik5 = [i for i in zip(p_score5, char5[0])]
        if weighted_plutchik5 != [] and char5[1] != [] and char5[2] != []:
            logger.info('char5: plutchik:{}, maslow:{}, reiss:{}'.format(weighted_plutchik5, char5[1], char5[2]))

        logger.info('x{}: {}'.format(i, x_sent))
        logger.info('context{}: {}'.format(i, con_sent))
        logger.info('y{}: {}'.format(i, y_sent))
        logger.info('pre_sent{}: {}'.format(i, " ".join(trans)))
        # cumulative BLEU scores

        # reference = [['this', 'is', 'small', 'test']]
        # candidate = ['this', 'is', 'a', 'test']

        # logger.info('Bleu_1: %f' % Bleu_1)
        # logger.info('Bleu_2: %f' % Bleu_2)
        # logger.info('Bleu_3: %f' % Bleu_3)
        # logger.info('Bleu_4: %f' % Bleu_4)
        # logger.info('rouge_1: %f' % rouge_1)
        # logger.info('rouge_2: %f' % rouge_2)
        # logger.info('rouge_l: %f' % rouge_l)
        # logger.info('acer_score: %f' % acer_score + '\n')
    '''
    # '''
    logger.info('x{}: {}'.format(i, x_sent))
    # logger.info('context{}: {}'.format(i, con_sent))
    # logger.info('y{}: {}'.format(i, y_sent))
    logger.info('pre_sent{}: {}'.format(i, " ".join(trans)))
    # '''
    return b1, b2, b3, b4, r1, r2, rl, aser


def test(args, model, clf_model, p_np, m_np, r_np, dev_loader, inv_dict, word_dict, device):
    x_sent = []
    con_sent = []
    y_sent = []
    p_score = []
    m_score = []
    r_score = []
    char_idx = []
    ref_file = args.ref_file
    hypo_file = args.hypo_file
    b1 = 0
    b2 = 0
    b3 = 0
    b4 = 0
    r1, r2, rl = 0, 0, 0
    aser = 0
    ref_writer = open(ref_file, 'w')
    hypo_writer = open(hypo_file, 'w')
    # ref_writer = None
    # hypo_writer = None
    for data in dev_loader:
        x_sent.append(data[0])
        con_sent.append(data[2])
        char_idx.append(data[4])
        p_score.append(data[5])
        m_score.append(data[6])
        r_score.append(data[7])
        y_sent.append(data[8])

    x_sent1 = x_sent
    con_sent1 = con_sent
    # 训练
    logger.info('testing...')
    if args.self_test:
        if args.only_plutchik:
            x_sent = 'Jane loves Marry .'.lower()
            con_sent = 'eos bos'.lower()
            p_pre = [[0, 0, 0, 0, 0, 0]]
            p_next = [[0, 0, 0, 0, 0, 0]]  # 指定接下来下来情感
            pmr_score = [[[0]]]
            char_idx = [[word_dict['i'], word_dict['you'], word_dict['none'], word_dict['none'], word_dict['none'],
                         word_dict['none']]]
            y_sent = ''
        else:
            while True:
                x_sent = input('input sentence: ').lower()
                # x_sent = 'Jane bought a new necklace .'.lower()
                con_sent = 'eos bos'.lower()
                char_num = eval(input('input character number: '))
                # p_inp = input('input plutchik: ').lower()
                idx = 0
                j = 0
                for i in range(4):
                    p_score = [[[0]*8, [0]*8, [0]*8, [0]*8, [0]*8, [0]*8]]
                    m_score = [[[0]*5, [0]*5, [0]*5, [0]*5, [0]*5, [0]*5]]
                    r_score = [[[0]*19, [0]*19, [0]*19, [0]*19, [0]*19, [0]*19]]
                    for i in range(char_num):
                        while True:
                            inp = input('input pmr of char{} : '.format(i)).lower().split(' ')
                            if inp == ['']:
                                break
                            if inp[1] in plutchik:
                                p_score[0][i][plutchik.index(inp[1])] = eval(inp[0])
                            elif inp[1] in maslow:
                                m_score[0][i][maslow.index(inp[1])] = eval(inp[0])
                            elif inp[1] in reiss:
                                r_score[0][i][reiss.index(inp[1])] = eval(inp[0])
                            else:
                                print('You have wrong input, please input again.')

                    char_idx = [[word_dict['i'], word_dict['you'], word_dict['none'], word_dict['none'], word_dict['none'],
                                 word_dict['none']]]
                    y_sent = ''
                    x_sent, con_sent = translate_story_dev(idx, j, args, word_dict, model, p_np, m_np, r_np, x_sent,
                                                           con_sent, y_sent, char_idx, inv_dict, p_score, m_score,
                                                           r_score, device)
                    idx += 1
                    if idx == 3:
                        break
        # b1, b2, b3, b4, r1, r2, rl, aser = translate_dev(0, args, word_dict, model, clf_model, p_np, m_np, r_np, x_sent,
        #                  con_sent, y_sent, char_idx, inv_dict,
        #                  p_score, m_score, r_score, device, b1, b2, b3, b4, r1,
        #                  r2, rl, aser, ref_writer, hypo_writer)

    else:
        if args.test_story:  # 测试故事
            ######################
            # 在con_sent中寻找只有[3, 4]的索引，这就是应该遍历的句子索引
            conlist = []
            for idx, i in enumerate(con_sent):
                if len(i) == 2:
                    conlist.append(idx)
            for idx1, i in enumerate(conlist):
                if idx1 < 10:
                    for idx, j in enumerate(range(i, i + 5)):
                        x_sent, con_sent = translate_story_dev(idx, j, args, word_dict, model, p_np, m_np, r_np, x_sent,
                                                               con_sent, y_sent, char_idx, inv_dict, p_score, m_score,
                                                               r_score, device)
                    x_sent = x_sent1
                    con_sent = con_sent1
            sys.exit(0)
            #######################
            # for idx, i in enumerate(range(6, 6+5)):
            #     x_sent, con_sent = translate_story_dev(idx, i, args, word_dict, model, p_np, m_np, r_np, x_sent, con_sent, y_sent,
            #                                  char_idx, inv_dict, p_score, m_score, r_score, device)
        else:
            # b1 = 0
            # b2 = 0
            # b3 = 0
            # b4 = 0
            # r1, r2, rl = 0, 0, 0
            # aser = 0
            for i in tqdm(range(len(dev_loader))):
                if i > 999999:
                    break
                # for i in tqdm(range(5)):
                b1, b2, b3, b4, r1, r2, rl, aser = translate_dev(i, args, word_dict, model, clf_model, p_np, m_np, r_np, x_sent,
                                                                 con_sent, y_sent, char_idx, inv_dict,
                                                                 p_score, m_score, r_score, device, b1, b2, b3, b4, r1,
                                                                 r2, rl, aser, ref_writer, hypo_writer)
                # translate_dev(i, args, word_dict, model, p_np, m_np, r_np, x_sent,
                #                                            con_sent, y_sent, char_idx, inv_dict,
                #                                            p_score, m_score, r_score, device, b1, b2, b3, b4, r1,
                #                                            r2, rl, ref_writer, hypo_writer)
            ref_writer.close()
            hypo_writer.close()
            logger.info('Average Bleu_1: %f' % (b1 / len(dev_loader)))
            logger.info('Average Bleu_2: %f' % (b2 / len(dev_loader)))
            logger.info('Average Bleu_3: %f' % (b3 / len(dev_loader)))
            logger.info('Average Bleu_4: %f' % (b4 / len(dev_loader)))
            logger.info('Average rouge_1: %f' % (r1 / len(dev_loader)))
            logger.info('Average rouge_2: %f' % (r2 / len(dev_loader)))
            logger.info('Average rouge_l: %f' % (rl / len(dev_loader)))
            logger.info('Average aser_score: %f' % (aser / len(dev_loader)))
            sys.exit(0)
            # pre_file = 'data/story_test.csv'
            # with open(pre_file, 'w', newline='') as csvfile:
            #     data = csv.writer(csvfile)
            #     for i in range(len(pre_list)):
            #         data.writerow([pre_list[i], y_list[i]])
            # print('写入完毕')


def main():
    # 读取命令行参数
    config_file = 'config/story_config.ini'

    switch = ['server_seq2seq_fix_indep', 'testing', '5_28']  # pro_all_data_5_2.csv

    parser = argparse.ArgumentParser()
    config = ConfigParser()
    config.read(config_file)

    # ref_file = 'data/ref.txt'
    # hypo_file = 'data/hypo.txt'
    parser.add_argument('--ref_file', default='result/ref_' + switch[0] + '_' + switch[2] + '.txt', help='self_test')
    parser.add_argument('--hypo_file', default='result/hypo_' + switch[0] + '_' + switch[2] + '.txt', help='self_test')
    parser.add_argument('--self_test', default=True, help='self_test')
    parser.add_argument('--test_story', default=False, help='self_test')
    parser.add_argument('--config_file', default=config_file, type=str, help='Select cuda number')
    parser.add_argument('--switch', default=switch, type=str, help='Select cuda number')
    parser.add_argument('--use_cuda', default=config.getboolean(switch[0], 'use_cuda'), type=str,
                        help='Select cuda number')
    parser.add_argument('--device', default=config.get(switch[0], 'device'), type=str, help='Select cuda number')
    parser.add_argument('--gpu_para', action='store_true', default=config.getboolean(switch[0], 'gpu_para'),
                        help='Whether load checkpoint')  # gpu parallel

    parser.add_argument('--log_path', default=config.get(switch[0], 'log_path').format(switch[1], switch[0], switch[2]),
                        type=str, required=False,
                        help='训练日志存放位置')
    parser.add_argument('--data_path', default=config.get(switch[0], 'data_path'),
                        help='load data file path')  # train_sen_char_idx / train_gpt2_idx_12_24 / train_plutchik_12_26
    parser.add_argument('--raw_data_path', default=config.get(switch[0], 'raw_data_path'), help='load data file path')
    parser.add_argument('--num_epochs', type=int, default=config.getint(switch[0], 'num_epochs'), help='num_epochs')
    parser.add_argument('--seed', type=int, default=config.getint(switch[0], 'seed'),
                        help='设置种子用于生成随机数，以使得训练的结果是确定的')  # None
    parser.add_argument('--batch_size', type=int, default=config.getint(switch[0], 'batch_size'),
                        help='number of batch_size')  # batch_size
    parser.add_argument('--num_workers', type=int, default=config.getint(switch[0], 'num_workers'),
                        help='number of workers')
    parser.add_argument('--lr', type=float, default=config.getfloat(switch[0], 'lr'), help='size of learning rate')
    parser.add_argument('--dropout', type=float, default=config.getfloat(switch[0], 'dropout'), help='size of dropout')
    parser.add_argument('--max_grad_norm', type=float, default=config.getfloat(switch[0], 'max_grad_norm'),
                        help='size of dropout')  # 1. / 5.
    parser.add_argument('--embedding_dim', type=int, default=config.getint(switch[0], 'embedding_dim'),
                        help='embedding_dim')  # 128 / 768 / 50 / 300
    parser.add_argument('--hidden_size', type=int, default=config.getint(switch[0], 'hidden_size'), help='hidden_size')
    parser.add_argument('--max_oovs', type=int, default=config.getint(switch[0], 'max_oovs'), help='number of max_oovs')
    parser.add_argument('--char_num', type=int, default=config.getint(switch[0], 'char_num'),
                        help='number of character')
    parser.add_argument('--pmr_size', type=int, default=config.getint(switch[0], 'pmr_size'), help='number of pmr')
    parser.add_argument('--p_size', type=int, default=config.getint(switch[0], 'p_size'), help='number of plutchik')
    parser.add_argument('--m_size', type=int, default=config.getint(switch[0], 'm_size'), help='number of maslow')
    parser.add_argument('--r_size', type=int, default=config.getint(switch[0], 'r_size'), help='number of reiss')
    parser.add_argument('--embed', default=config.get(switch[0], 'embed'),
                        help='Select 50d or 300d embedding file path')
    parser.add_argument('--word_dict', default=config.get(switch[0], 'word_dict'), help='Select word_dict file path')
    parser.add_argument('--glove', action='store_true', default=config.getboolean(switch[0], 'glove'),
                        help='Whether use glove')

    # model
    parser.add_argument('--adjust_lr', action='store_true', default=config.getboolean(switch[0], 'adjust_lr'),
                        help='adjust_lr')
    parser.add_argument('--opt', action='store_true', default=config.getboolean(switch[0], 'opt'),
                        help='Select Adam or SGD optimizer. True is Adam')
    parser.add_argument('--bigru', action='store_true', default=config.getboolean(switch[0], 'bigru'),
                        help='Whether use bigru')
    parser.add_argument('--bilstm', action='store_true', default=config.getboolean(switch[0], 'bilstm'),
                        help='Whether use bilstm')
    parser.add_argument('--gate', action='store_true', default=config.getboolean(switch[0], 'gate'),
                        help='Whether use gate mechanism')
    parser.add_argument('--copy', action='store_true', default=config.getboolean(switch[0], 'copy'),
                        help='Whether use copy mechanism')
    parser.add_argument('--teacher_force', action='store_true', default=config.getboolean(switch[0], 'teacher_force'),
                        help='Whether use teacher force')

    # pmr & char
    parser.add_argument('--self_attention', action='store_true', default=config.getboolean(switch[0], 'self_attention'))
    parser.add_argument('--fix_encoder', action='store_true', default=config.getboolean(switch[0], 'fix_encoder'))
    parser.add_argument('--encoder_merge', action='store_true', default=config.getboolean(switch[0], 'encoder_merge'))
    parser.add_argument('--baseline', action='store_true', default=config.getboolean(switch[0], 'baseline'))
    parser.add_argument('--fix_decoder', action='store_true', default=config.getboolean(switch[0], 'fix_decoder'))
    parser.add_argument('--psy_clf', action='store_true', default=config.getboolean(switch[0], 'psy_clf'))
    parser.add_argument('--seq_attn', action='store_true', default=config.getboolean(switch[0], 'seq_attn'))
    parser.add_argument('--context', action='store_true', default=config.getboolean(switch[0], 'context'),
                        help='Whether add context')
    parser.add_argument('--only_plutchik', action='store_true', default=config.getboolean(switch[0], 'only_plutchik'),
                        help='Whether add pmr_input')
    parser.add_argument('--dynamic', action='store_true', default=config.getboolean(switch[0], 'dynamic'),
                        help='Whether add pmr_input')
    parser.add_argument('--pmr_input', action='store_true', default=config.getboolean(switch[0], 'pmr_input'),
                        help='Whether add pmr_input')
    parser.add_argument('--rep_inp_attn', action='store_true', default=config.getboolean(switch[0], 'rep_inp_attn'),
                        help='Whether add rep_inp_attn')
    parser.add_argument('--pmr_attn', action='store_true', default=config.getboolean(switch[0], 'pmr_attn'),
                        help='Whether use pmr_attn')
    parser.add_argument('--char_attn', action='store_true', default=config.getboolean(switch[0], 'char_attn'),
                        help='Whether use char_attn')

    # load & save model
    parser.add_argument('--load_ckpt', action='store_true', default=config.getboolean(switch[0], 'load_ckpt'),
                        help='Whether load checkpoint')  # load checkpoint
    parser.add_argument('--save_ckpt', action='store_true', default=config.getboolean(switch[0], 'save_ckpt'),
                        help='Whether save checkpoint')  # save checkpoint
    parser.add_argument('--load_ckpt_file',
                        default=config.get(switch[0], 'load_ckpt_file').format(switch[2], switch[0]),
                        help='Set checkpoint file path')  # ckpt_path
    parser.add_argument('--ckpt_path', default=config.get(switch[0], 'ckpt_path').format(switch[2]),
                        help='Set checkpoint file path')  # ckpt_path
    parser.add_argument('--ckpt_file', default=config.get(switch[0], 'ckpt_file').format(switch[0]),
                        help='Set checkpoint file name')  # ckpt_file
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    global logger
    logger = create_logger(args)
    logger.info('start game!')
    logger.info('switch: {}'.format(switch))
    logger.info(args)

    if args.use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device('cpu')
    logger.info(device)

    if args.gpt2 == False:
        # 加载word_dict
        with open(args.word_dict) as f:
            for line in f:
                word_dict = json.loads(line)

        p_np, m_np, r_np = get_pmr(args, word_dict)  # (32, 1)

        vocab_size = len(word_dict)
        args.vocab_size = vocab_size

    embedding_matrix = None
    # model
    attention = Attention(args.hidden_size)
    encoder = Encoder(args, device, embedding_matrix, vocab_size, word_dict, args.embedding_dim, args.hidden_size,
                      dropout=args.dropout)
    decoder = Decoder(args, attention, encoder, word_dict, vocab_size, args.embedding_dim, args.char_num,
                      args.hidden_size, device,
                      dropout=args.dropout)
    model = PreStory(args, word_dict, device, encoder, decoder, args.embedding_dim, args.pmr_size,
                     args.hidden_size)

    clf_model = PMRClf(args, device, embedding_matrix, encoder, vocab_size, word_dict, args.embedding_dim, args.hidden_size,
                   dropout=args.dropout).to(device)

    if args.use_cuda and args.gpu_para:
        # model = nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
        model = nn.DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])  # multi-GPU
        torch.backends.cudnn.benchmark = True
    model = model.to(device)

    model.load_state_dict(torch.load(args.load_ckpt_file + '.pkl'))
    logger.info('loading checkpoint file {}'.format(args.load_ckpt_file))
    clf_model=None
    '''
    save_model = torch.load('checkpoint/params_server_pmr_clf_16epoch.pkl')
    # save_model = torch.load('checkpoint/6_2/params_server_pmr_clf_13epoch.pkl')
    model_dict = clf_model.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    # clf_model.load_state_dict(torch.load('checkpoint/params_server_pmr_clf_16epoch.pkl'))
    logger.info('loading clf_model checkpoint file {}'.format('params_server_pmr_clf_16epoch.pkl'))
    '''

    dataset = MyDataset_pmr(args.data_path)  # dataset.__len__(): 19412

    train_loader, dev_loader = train_test_split(dataset, test_size=0.1, random_state=1)
    logger.info("loading {} data".format('dev_loader'))
    inv_dict = {v: k for k, v in word_dict.items()}
    args.inv_dict = inv_dict
    test(args, model, clf_model, p_np, m_np, r_np, dev_loader, inv_dict, word_dict, device)


if __name__ == '__main__':
    main()
