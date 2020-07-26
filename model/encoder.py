import torch
import os
import torch.nn as nn
import torch.nn.functional as F
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


class Encoder(nn.Module):
    def __init__(self, args, device, embedding_matrix, vocab_size, word_dict, embedding_dim, hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.args = args
        self.word_dict = word_dict
        idx2word = {v: k for k, v in word_dict.items()}
        glove_path = 'data/glove_word2vec_300d.txt'
        if self.args.glove:
            # self.embed = None
            self.embed = nn.Embedding.from_pretrained(
                get_pretrained_glove(path=glove_path, idx2word=idx2word, n_special=len(token)),
                freeze=False, padding_idx=0)  # specials: pad, unk, naf_h/t
        else:
            self.embed = nn.Embedding(vocab_size, embedding_dim)

        # self.pmr_init_matrix = torch.randn(args.char_num*args.pmr_size, args.hidden_size*2).to(device)  # (6*32, 2h) randomly initialize a pmr matrix
        # 2-17
        # self.pmr_init_matrix = torch.randn(args.pmr_size, args.hidden_size*2).to(device)  # (32, 2h) randomly initialize a pmr matrix
        # 2-28
        self.pmr_init_matrix = nn.Parameter(torch.randn(args.pmr_size, args.hidden_size * 2),
                                            requires_grad=True)  # (32, 2h) randomly initialize a pmr matrix

        self.Wh1 = nn.Linear(hidden_size * 1, hidden_size * 2)
        self.Wc1 = nn.Linear(hidden_size * 1, hidden_size * 2)

        self.bilstm = nn.LSTM(embedding_dim, hidden_size * 2, 2, batch_first=True, bidirectional=True)

        self.bilstm1 = nn.LSTM(embedding_dim, hidden_size * 2, 2, batch_first=True, bidirectional=True)

        self.dropout_x = nn.Dropout(dropout)
        self.dropout_c = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_hid = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_hid1 = nn.Linear(hidden_size * 4, hidden_size)
        # self.fc_hid_lstm = nn.Linear(hidden_size * 2, hidden_size)
        # self.fc_hid_lstm1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_c1 = nn.Linear(hidden_size * 4, hidden_size)
        # self.fc_c_lstm = nn.Linear(hidden_size * 2, hidden_size)
        # self.fc_c_lstm1 = nn.Linear(hidden_size * 2, hidden_size)
        self.enc_out = nn.Linear(hidden_size * 4, hidden_size * 2)
        # self.enc_out_lstm = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.enc_out_lstm1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.W_s1 = nn.Linear(2 * hidden_size, 350)
        self.W_s2 = nn.Linear(350, 30)
        self.fc_layer = nn.Linear(30 * 2 * hidden_size, hidden_size * 2)

    def encode_sen(self, x, x_len):
        sorted_len, sorted_idx = x_len.sort(0, descending=True)  # 把batch里的sequence按长度排序（从长到短）
        x_sorted = x[sorted_idx.long()]  # 句子已经按长度排序
        embedded = self.dropout_x(self.embed(x_sorted))  # (b,max_len,e)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            # (sum_len,e)
                                                            batch_first=True)  # 将不定长的sequence最后的pad部分pack掉
        if self.args.bilstm:
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
            # hid = self.fc(hid).unsqueeze(0)  # (1, batch_size, hidden_size)
            hid = torch.tanh(self.fc_hid(hid)).unsqueeze(0)  # (1, batch_size, hidden_size)
            c = torch.cat([c[-2], c[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
            # c = self.fc(c).unsqueeze(0)  # (1, batch_size, hidden_size)
            c = torch.tanh(self.fc_c(c)).unsqueeze(0)  # (1, batch_size, hidden_size)

            # out:(batch_size, max_len, hidden_size*2)， hid:(1, batch_size, hidden_size)
            # return out, hid[[-1]]
            out = self.enc_out(out)
            return out, hid, c  # enc_char, rep

    def encode_con(self, x, x_len):
        sorted_len, sorted_idx = x_len.sort(0, descending=True)  # 把batch里的sequence按长度排序（从长到短）
        x_sorted = x[sorted_idx.long()]  # 句子已经按长度排序
        embedded = self.dropout_c(self.embed(x_sorted))  # (b,max_len,e)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            # (sum_len,e)
                                                            batch_first=True)  # 将不定长的sequence最后的pad部分pack掉
        if self.args.bilstm:
            self.bilstm.flatten_parameters()
            # hid为除去句子最后pad的部分，句子实际长度最后的hidden.  hid:(num_layers2, batch_size, hidden_size*2)
            packed_out, (hid, c) = self.bilstm1(packed_embedded.float())
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(
                1))  # 将pack掉的sequence最后再pad #out:(batch_size, max_len, hidden_size*2)
            _, original_idx = sorted_idx.sort(0, descending=False)  # 将句子变为原来的顺序
            out = out[original_idx.long()].contiguous()  # 将句子变为原来的顺序
            hid = hid[:,
                  original_idx.long()].contiguous()  # hid排序 hid：(num_layers * num_directions, batch_size, hidden_size)
            # add attention
            hid = torch.cat([hid[-2], hid[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
            # hid = self.fc(hid).unsqueeze(0)  # (1, batch_size, hidden_size)
            hid = torch.tanh(self.fc_hid1(hid)).unsqueeze(0)  # (1, batch_size, hidden_size)
            c = torch.cat([c[-2], c[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
            # c = self.fc(c).unsqueeze(0)  # (1, batch_size, hidden_size)
            c = torch.tanh(self.fc_c1(c)).unsqueeze(0)  # (1, batch_size, hidden_size)

            # out:(batch_size, max_len, hidden_size*2)， hid:(1, batch_size, hidden_size)
            out = self.enc_out(out)
            return out, hid, c  # enc_char, rep

    def attention_net(self, lstm_output):

        """
    Now we will use self attention mechanism to produce a matrix embedding of the input sentence in which every row represents an
    encoding of the inout sentence but giving an attention to a specific part of the sentence. We will use 30 such embedding of
    the input sentence and then finally we will concatenate all the 30 sentence embedding vectors and connect it to a fully
    connected layer of size 2000 which will be connected to the output layer of size 2 returning logits for our two classes i.e.,
    pos & neg.
    Arguments
    ---------
    lstm_output = A tensor containing hidden states corresponding to each time step of the LSTM network.
    ---------
    Returns : Final Attention weight matrix for all the 30 different sentence embedding in which each of 30 embeddings give
              attention to different parts of the input sentence.
    Tensor size : lstm_output.size() = (batch_size, num_seq, 2*hidden_size)
                  attn_weight_matrix.size() = (batch_size, 30, num_seq)
    """
        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix

    def forward(self, sentence, sentence_len, context, context_len, char_idx, p_id, m_id, r_id,
                p_score, m_score, r_score):
        '''
        char_idx:   (b, 6)
        p_score:  (b, 6, 8)
        m_score:  (b, 6, 5)
        r_score:  (b, 6, 19)
        p_id:        (8, 1)
        m_id:        (5, 1)
        r_id:        (19, 1)
        '''

        batch_size = sentence.size(0)
        if self.args.fix_encoder:
            # print(self.embed.weight)
            enc_char, p_matrix, m_matrix, r_matrix = None, None, None, None
            # 处理pmr_matrix
            pmr_score = torch.cat([p_score, m_score, r_score], 2).view(batch_size, -1).unsqueeze(2)  # (b, 6*32, 1)
            # pmr_matrix = self.pmr_init_matrix.repeat(batch_size, 1, 1) * pmr_score  # (b, 6*32, 1) * (b, 6*32, 2h) = (b, 6*32, 2h)
            # 2-17
            pmr_matrix = self.pmr_init_matrix.repeat(batch_size, self.args.char_num,
                                                     1) * pmr_score  # (b, 6*32, 2h) * (b, 6*32, 1) = (b, 6*32, 2h)
            # 处理sentence, context
            # 合并经过一个LSTM
            if self.args.context:
                if self.args.encoder_merge:
                    selfattn_out = None
                    # [bos + context + sentence + eos]
                    sentence = sentence[sentence != self.word_dict['bos']].view(
                        sentence.size(0), -1)
                    context = context[context != self.word_dict['eos']].view(
                        context.size(0), -1)
                    x = torch.cat([context, sentence], 1)
                    x_len = sentence_len + context_len - 2
                    out, hid, c = self.encode_sen(x, x_len)
                    if self.args.self_attention:
                        attn_weight_matrix = self.attention_net(out)
                        hidden_matrix = torch.bmm(attn_weight_matrix, out)
                        selfattn_out = self.fc_layer(
                            hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])).unsqueeze(
                            1)  # (b, 1, 2h)
                    hid = self.Wh1(hid)  # (1,b,2h)
                    c = self.Wc1(c)
                    return out, hid, c, enc_char, pmr_matrix, selfattn_out, r_matrix
                # 分别经过LSTM
                else:
                    # [bos + sentence + eos]
                    # [bos + context + eos]
                    selfattn_out_sen, selfattn_out_con = None, None
                    sen_out, sen_hid, sen_c = self.encode_sen(sentence, sentence_len)
                    con_out, con_hid, con_c = self.encode_con(context, context_len)
                    if self.args.self_attention:
                        attn_weight_matrix_sen = self.attention_net(sen_out)
                        hidden_matrix_sen = torch.bmm(attn_weight_matrix_sen, sen_out)
                        selfattn_out_sen = self.fc_layer(
                            hidden_matrix_sen.view(-1, hidden_matrix_sen.size()[1] * hidden_matrix_sen.size()[
                                2])).unsqueeze(
                            1)  # (b, 1, 2h)
                        attn_weight_matrix_con = self.attention_net(con_out)
                        hidden_matrix_con = torch.bmm(attn_weight_matrix_con, con_out)
                        selfattn_out_con = self.fc_layer(
                            hidden_matrix_con.view(-1, hidden_matrix_con.size()[1] * hidden_matrix_con.size()[
                                2])).unsqueeze(
                            1)  # (b, 1, 2h)

                    out = torch.cat((con_out, sen_out), 1)  # (b, sen_len+con_len, 2h)
                    hid = torch.cat((con_hid, sen_hid), 2)  # (1, b, 2h)
                    c = torch.cat((con_c, sen_c), 2)  # (1, b, 2h)
                    return sen_out, hid, c, con_out, pmr_matrix, selfattn_out_sen, selfattn_out_con
            else:
                sen_out, sen_hid, sen_c = self.encode_sen(sentence, sentence_len)
                # sen_out:(b,seq,2h) sen_hid:(1,b,h) sen_c:(1,b,h)
                hid = self.Wh1(sen_hid)  # (1,b,2h)
                c = self.Wc1(sen_c)
                return sen_out, hid, c, enc_char, pmr_matrix, m_matrix, r_matrix
