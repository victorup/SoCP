from torch.functional import F
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, out_size, dropout=0.2):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        # self.embed = nn.Embedding()

        self.Wchar = nn.Linear(embedding_dim, hidden_size)

        self.v_c = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.Wh = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        self.bilstm = nn.LSTM(embedding_dim, hidden_size * 2, 2, batch_first=True, bidirectional=True)
        self.bilstm1 = nn.LSTM(embedding_dim, hidden_size * 2, 2, batch_first=True, bidirectional=True)

        self.dropout_e = nn.Dropout(dropout)
        self.dropout_c = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_hid = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_hid1 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_c = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_c1 = nn.Linear(hidden_size * 4, hidden_size)
        self.fc_out_p = nn.Linear(hidden_size * 2, out_size['p'])
        self.fc_out_m = nn.Linear(hidden_size * 2, out_size['m'])
        self.fc_out_r = nn.Linear(hidden_size * 2, out_size['r'])
        self.enc_out = nn.Linear(hidden_size * 4, hidden_size * 2)

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
        # hid = self.fc(hid).unsqueeze(0)  # (1, batch_size, hidden_size)
        hid = torch.tanh(self.fc_hid(hid)).unsqueeze(0)  # (1, batch_size, hidden_size)
        c = torch.cat([c[-2], c[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
        # c = self.fc(c).unsqueeze(0)  # (1, batch_size, hidden_size)
        c = torch.tanh(self.fc_c(c)).unsqueeze(0)  # (1, batch_size, hidden_size)

        # out:(batch_size, max_len, hidden_size*2)， hid:(1, batch_size, hidden_size)
        out = self.enc_out(out)
        return out, hid, c  # enc_char, rep

    def encode_con(self, x, x_len):
        sorted_len, sorted_idx = x_len.sort(0, descending=True)  # 把batch里的sequence按长度排序（从长到短）
        x_sorted = x[sorted_idx.long()]  # 句子已经按长度排序
        embedded = self.dropout_e(self.embed(x_sorted))  # (b,max_len,e)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_len.long().cpu().data.numpy(),
                                                            # (sum_len,e)
                                                            batch_first=True)  # 将不定长的sequence最后的pad部分pack掉
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
        hid = torch.tanh(self.fc_hid1(hid)).unsqueeze(0)  # (1, batch_size, hidden_size)
        c = torch.cat([c[-2], c[-1]], dim=1)  # (batch_size, hidden_size+hidden_size)
        c = torch.tanh(self.fc_c1(c)).unsqueeze(0)  # (1, batch_size, hidden_size)

        # out:(batch_size, max_len, hidden_size*2)， hid:(1, batch_size, hidden_size)
        out = self.enc_out(out)
        return out, hid, c  # enc_char, rep

    def attention(self, char_embed_matrix, batch_size, hid, ):
        att = self.v_c(torch.tanh(
            self.Wchar(char_embed_matrix.contiguous().view(-1, char_embed_matrix.size(2))) + self.Wh(
                hid.squeeze(0))))  # (b*6,2h) + (b*6,2h) --> (b*6,1)
        # print(att.size())
        # print(hid.size())
        attn_score = F.softmax(att.view(batch_size, hid.squeeze(0).size()[-1]), dim=1)  # (b, 1)
        # char_attn = torch.bmm(attn_score.unsqueeze(0), hid)  # [b x 1 x 6] * [b x 6 x hidden*2]
        char_attn = attn_score.unsqueeze(0) * hid
        char_attn = char_attn.squeeze(1)  # [x b hidden*2]
        return char_attn

    def forward(self, sentence, sentence_len, context, context_len, char_idx):
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
        # 处理sentence, context
        embedded_char_idx = self.dropout_c(self.embed(char_idx))
        sen_out, sen_hid, sen_c = self.encode_sen(sentence, sentence_len)
        con_out, con_hid, con_c = self.encode_con(context, context_len)
        c = torch.cat((sen_c, con_c), 2)  # (1, b, 2h)
        c_att = self.attention(embedded_char_idx, batch_size, c)  # (b, 2h)
        p = self.fc_out_p(c_att)
        m = self.fc_out_m(c_att)
        r = self.fc_out_r(c_att)
        return p, m, r
