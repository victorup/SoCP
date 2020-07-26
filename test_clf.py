import torch
import argparse
import torch.nn as nn
import json
from configparser import ConfigParser
from gensim.models import KeyedVectors
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from packages.function import create_logger
from dataloder import MyDataset_clf, get_pmr
from packages.acer import ACER
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys
from model.attention import Attention
from model.pmr_clf import PMRClf

from model.encoder import Encoder
from model.decoder import Decoder
from model.story_seq2seq import PreStory

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

    attention = attention.squeeze(0).cpu().detach().numpy()
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


def translate_dev(i, args, model, x, y, inv_dict, score, device):
    x_sent = " ".join([inv_dict[w] for w in x[i]]).replace('bos', '').replace('eos', '').strip()

    mini_x = torch.tensor(x[i]).long().to(device).unsqueeze(0)  # (b, len)
    mini_x_len = torch.tensor(len(x[i])).long().to(device).unsqueeze(0)  # (b)
    mini_y = torch.tensor(y[i]).long().to(device).unsqueeze(0)  # (b,6)

    if args.gpu_para and args.use_cuda:
        pre_y = model.module.predict_p(mini_x, mini_x_len, mini_y)
    else:
        pre_y = model.predict_p(mini_x, mini_x_len, mini_y)
    if pre_y.item() == mini_y.item():
        score += 1
    logger.info('x{}: {}'.format(i, x_sent))
    if pre_y.item() == 0:
        logger.info('y{}: {}'.format(i, 'none'))
    else:
        logger.info('y{}: {}'.format(i, plutchik[pre_y.item()-1]))


def test(args, model, dev_loader, inv_dict, word_dict, device):
    x = []
    y = []
    score = 0
    for data in dev_loader:
        x.append(data[0])
        y.append(data[2])

    # 训练
    logger.info('testing...')
    for i in tqdm(range(len(dev_loader))):
        translate_dev(i, args, model, x, y, inv_dict, score, device)
    acc = score / len(dev_loader)
    logger.info('acc: {}'.format(acc))


def main():
    # 读取命令行参数
    config_file = 'config/story_config.ini'

    switch = ['server_pmr_clf', 'testing', '2_17']

    parser = argparse.ArgumentParser()
    config = ConfigParser()
    config.read(config_file)

    parser.add_argument('--ref_file', default='result/ref_' + switch[0] + '_' + switch[2] + '.txt', help='self_test')
    parser.add_argument('--hypo_file', default='result/hypo_' + switch[0] + '_' + switch[2] + '.txt', help='self_test')
    parser.add_argument('--self_test', default=False, help='self_test')
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
    parser.add_argument('--opt', action='store_true', default=config.getboolean(switch[0], 'opt'),
                        help='Select Adam or SGD optimizer. True is Adam')
    parser.add_argument('--gpt2', action='store_true', default=config.getboolean(switch[0], 'gpt2'),
                        help='Whether use gpt2')
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
    # torch.backends.cudnn.deterministic = True

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

        if args.glove:
            # inv_dict = {v: k for k, v in word_dict.items()}
            # 加载glove预训练词向量
            logger.info('loading: {}'.format(args.embed))
            tmp_word2vec = args.embed
            glove_model = KeyedVectors.load_word2vec_format(tmp_word2vec)
            args.embedding_dim = glove_model.vector_size
            embedding_matrix = np.zeros((len(word_dict), args.embedding_dim))  # [vocab_size，embedding_dim]
            for i in range(len(word_dict)):
                embedding_matrix[i, :] = glove_model[glove_model.index2word[i]]
            embedding_matrix = torch.from_numpy(embedding_matrix).float().to(device)
        else:
            embedding_matrix = None

        p_np, m_np, r_np = get_pmr(args, word_dict)  # (32, 1)

        vocab_size = len(word_dict)

    # model
    model = PMRClf(args, device, embedding_matrix, vocab_size, word_dict, args.embedding_dim, args.hidden_size,
                      dropout=args.dropout)
    if args.use_cuda and args.gpu_para:
        # model = nn.DataParallel(model, device_ids=[0, 1])  # multi-GPU
        model = nn.DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])  # multi-GPU
        torch.backends.cudnn.benchmark = True
    model = model.to(device)

    model.load_state_dict(torch.load(args.load_ckpt_file + '.pkl'))
    logger.info('loading checkpoint file {}'.format(args.load_ckpt_file))

    dataset = MyDataset_clf(args.data_path)

    train_loader, dev_loader = train_test_split(dataset, test_size=0.1, random_state=1)
    logger.info("loading {} data".format('dev_loader'))
    inv_dict = {v: k for k, v in word_dict.items()}
    test(args, model, dev_loader, inv_dict, word_dict, device)


if __name__ == '__main__':
    main()
