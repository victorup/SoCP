import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

token = ['<pad>', '<none>', '<unk>', '<bos>', '<eos>']


def get_pretrained_glove(path, idx2word, n_special=5):
    saved_glove = path.replace('.txt', '.pt')

    def make_glove():
        print('Reading pretrained glove...')
        glove_dict = {}
        with open(path, 'r') as f:
            for idx, item in enumerate(f.readlines()):
                if idx == 0:
                    continue
                glove_dict[item.split(' ')[0].lower()] = np.array(item.split(' ')[1:])

        def get_vec(w):
            w = w.lower()
            try:
                return glove_dict[w].astype('float32')
            except KeyError:
                return np.zeros((300,), dtype='float32')

        weights = [torch.from_numpy(get_vec(w)) for i, w in list(idx2word.items())[n_special:]]
        weights = torch.stack(weights, dim=0)

        addvec = torch.randn(n_special, weights.size(1))
        weights = torch.cat([addvec, weights], dim=0)
        torch.save(weights, saved_glove)
        print(f"Glove saved in {saved_glove}")

    if not os.path.isfile(saved_glove):
        make_glove()
    print('load glove')
    return torch.load(saved_glove)


class PMRClf(nn.Module):
    def __init__(self, args, device, embedding_matrix, encoder, vocab_size, word_dict, embedding_dim, hidden_size, dropout=0.2):
        super(PMRClf, self).__init__()
        self.args = args
        self.word_dict = word_dict
        idx2word = {v: k for k, v in word_dict.items()}
        glove_path = 'data/glove_word2vec_300d.txt'
        self.embed = nn.Embedding.from_pretrained(
            get_pretrained_glove(path=glove_path, idx2word=idx2word, n_special=len(token)),
            freeze=False, padding_idx=0)  # specials: pad, unk, naf_h/t

        self.bilstm = nn.LSTM(embedding_dim, hidden_size * 2, 2, batch_first=True, bidirectional=True)
        self.dropout_e = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_hid = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 4, hidden_size)
        self.enc_out = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.out = nn.Linear(hidden_size, 128)
        self.out1 = nn.Linear(128, 9)

    def encode_sen(self, x, x_len):
        sorted_len, sorted_idx = x_len.sort(0, descending=True)  # 把batch里的sequence按长度排序（从长到短）
        x_sorted = x[sorted_idx.long()]  # 句子已经按长度排序
        embedded = self.dropout_e(self.embed(x_sorted))  # (b,max_len,e)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            # (sum_len,e)
                                                            batch_first=True)  # 将不定长的sequence最后的pad部分pack掉
        self.bilstm.flatten_parameters()
        # hid为除去句子最后pad的部分，句子实际长度最后的hidden.  hid:(num_layers2, batch_size, hidden_size*2)
        packed_out, (hid, c) = self.bilstm(packed_embedded.float())
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(
            1))  # 将pack掉的sequence最后再pad #out:(batch_size, max_len, hidden_size*2)
        _, original_idx = sorted_idx.sort(0, descending=False)  # 将句子变为原来的顺序
        out = out[original_idx.long()].contiguous()  # 将句子变为原来的顺序
        hid = hid[:,
              original_idx.long()].contiguous()  # hid排序 hid：(num_layers * num_directions, batch_size, hidden_size)
        # add attention
        hid = torch.cat([hid[-2], hid[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
        hid = torch.tanh(self.fc_hid(hid)).unsqueeze(0)  # (1, batch_size, hidden_size)
        c = torch.cat([c[-2], c[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
        c = torch.tanh(self.fc_c(c)).unsqueeze(0)  # (1, batch_size, hidden_size)

        # out:(batch_size, max_len, hidden_size*2)， hid:(1, batch_size, hidden_size)
        out = self.enc_out(out)
        return out, hid, c  # enc_char, rep

    def forward(self, sentence, sentence_len, y):
        out, hid, c = self.encode_sen(sentence, sentence_len)
        pre_y = self.out1(self.out(hid.squeeze(0)))
        return pre_y

    def predict_p(self, sentence, sentence_len, y):
        out, hid, c = self.encode_sen(sentence, sentence_len)  # hid:(1,b,2h)
        pre_y = F.softmax(self.out1(self.out(hid.squeeze(0))), 1)  # (b, 2h) --> (b, h) --> (b, 9)
        pre_y = pre_y.argmax()
        return pre_y
