# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

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


class Decoder(nn.Module):
    def __init__(self, args, attention, encoder, word_dict, vocab_size, embedding_dim, char_num, hidden_size,
                 device, dropout=0.2):
        super(Decoder, self).__init__()
        # self.tokenizer = tokenizer
        # self.gpt_model = gpt_model

        self.args = args
        self.attention = attention
        self.char_num = args.char_num
        self.pmr_size = args.pmr_size
        self.p_size = args.p_size
        self.m_size = args.m_size
        self.r_size = args.r_size
        self.word_dict = word_dict
        self.device = device
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        if args.glove:
            self.embed = encoder.embed
        self.y_inp = nn.Linear(embedding_dim + hidden_size * 2 + hidden_size * 2, embedding_dim)

        self.rnn2_attn = nn.LSTM(embedding_dim, hidden_size * 2, batch_first=True)

        self.out2 = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.Wchar = nn.Linear(embedding_dim + hidden_size*2 + hidden_size*2, char_num)

        # pmr attention
        self.fc_attn = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.w_p = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.w_h = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.v = nn.Linear(hidden_size * 2, 1, bias=False)

        self.fc_gate1 = nn.Linear(hidden_size * 4, hidden_size * 2)

    def init_hidden(self, batch_size):
        return torch.randn_like(torch.zeros(1, batch_size, self.hidden_size * 2)).to(self.device)

    def create_mask(self, src):
        mask = (src != self.word_dict['<pad>'])
        return mask

    def create_mask1(self, x_len, y_len):
        # a mask of shape x_len * y_len
        max_x_len = x_len.max()  # (x_max_len)
        max_y_len = y_len.max()  # (y_max_len)
        x_mask = torch.arange(max_x_len, device=x_len.device)[None, :] < x_len[:,
                                                                         None]  # (batch_size, x_max_len) None的维度上加1维. # 句子真实长度不够最大长度的为False，其余为True
        y_mask = torch.arange(max_y_len, device=x_len.device)[None, :] < y_len[:, None]
        mask = (~ x_mask[:, :, None] * y_mask[:, None, :]).bool()  # (batch_size, x_max_len, y_max_len) [1 0]
        return mask

    def to_cuda(self, tensor):
        # turns to cuda
        # if torch.cuda.is_available():
        if self.device.type != 'cpu':
            return tensor.cuda()
        else:
            return tensor

    def forward(self, input_idx, encoded_x, encoded_x_idx, encoded_con, encoded_con_idx,
                p_matrix, selfattn_out_sen, selfattn_out_con, hid, c, mask_x, mask_c, order):
        '''
        input_idx / y_(t-1): [b]			<- idx of next input to the decoder (Variable)
        encoded:             [b x seq x 2h]		<- hidden states created at encoder (Variable)
        encoded_idx:         [b x seq]       <- idx of inputs used at encoder (numpy)
        char_idx:            [b, 6]
        char_embed_matrix:   [b, 6, 2h]
        rep:        [b,6*32,2h]
        hid:                 [1,b,2h]
        c:                   [1,b,2h]
        attn:                [b x 1 x hidden*2]  <- weighted attention of previous state, init with all zeros (Variable)
        '''
        encoded, encoded_idx, char_embed_matrix = None, None, None
        p_attn_score, m_attn_score, r_attn_score, weighted = None, None, None, None
        if self.args.fix_decoder:
            batch_size = input_idx.size(0)
            # e(yt-1)
            y_input = self.dropout(self.embed(input_idx))  # (b, e) # (batch_size, embed_size)
            # seq_attn weighted
            if self.args.seq_attn:
                if self.args.self_attention and self.args.encoder_merge:
                    dec_attn = self.attention(hid.squeeze(0), encoded_x.transpose(0, 1), mask_x).unsqueeze(1)  # (b, 1, seq)
                    weighted = torch.bmm(dec_attn, encoded_x)  # (b, 1, 2h)
                else:
                    dec_attn_x = self.attention(hid.squeeze(0), encoded_x.transpose(0, 1), mask_x).unsqueeze(1)  # (b, 1, seq)
                    weighted_x = torch.bmm(dec_attn_x, encoded_x)  # (b, 1, 2h)
                    dec_attn_c = self.attention(hid.squeeze(0), encoded_con.transpose(0, 1), mask_c).unsqueeze(1)  # (b, 1, seq)
                    weighted_c = torch.bmm(dec_attn_c, encoded_con)  # (b, 1, 2h)
                    weighted = self.fc_gate1(torch.cat([weighted_x, weighted_c], 2))  # (4h->2h)
            if self.args.baseline:
                # y_in = torch.cat((y_input.unsqueeze(1), weighted), 2)  # (b,1,e+2h)
                y_in = y_input.unsqueeze(1)  # (b,1,e+2h)
                # lstm
                self.rnn2_attn.flatten_parameters()
                out, _ = self.rnn2_attn(y_in.float())  # out:(b, 1, h*2), hid:(1, b, h*2)  # out=hid
                out = F.softmax(self.out1(out), 2)  # (b,1,2h) --> (b,1,vocab_size)
                return out, hid, c, weighted, p_attn_score, m_attn_score, r_attn_score
                # return out
            else:
                '''
                p_matrix(pmr_matrix): (b, 6*32, 2h)
                '''
                # character selector
                g_char = F.softmax(
                    self.Wchar(
                        torch.cat([y_input, hid.squeeze(0), weighted.squeeze(1)], 1)  # (b, 1, e+2h+2h)
                    ), 1
                )  # (b,1,e+2h+2h) --> (b,6)
                z_char = g_char * torch.ne(p_matrix.view(batch_size, self.args.char_num, -1).sum(dim=2), 0)  # (b, 6)
                o_char = torch.zeros_like(z_char, requires_grad=True).scatter(1,
                                                                              torch.max(z_char, dim=1)[1].unsqueeze(1),
                                                                              1).to(self.device)
                s_char = (o_char.unsqueeze(2) * p_matrix.view(batch_size, 6, -1)).view(
                    batch_size, self.char_num*self.pmr_size, -1)  # (b,6,1) * (b, 6, 32*2h) = (b,6,32*2h)-->(b,6*32,2h)

                # psychological state controller
                att = self.v(torch.tanh(self.w_h(hid.squeeze(0).repeat(self.args.char_num*self.args.pmr_size, 1)) +
                                        self.w_p(s_char).contiguous().view(-1, s_char.size(2))))  # (b*6*32,2h)
                p_attn_score = F.softmax(att.view(batch_size, -1), dim=1)  # (b, 6*32)
                p_attn = torch.bmm(p_attn_score.unsqueeze(1), s_char)  # [b x 1 x 6*32] * [b x 6*32 x hidden*2]
                p_attn_score = p_attn_score.view(6, -1)  # (6,32)
                p_attn_score = p_attn_score[torch.arange(p_attn_score.size(0)) == o_char.squeeze().argmax().item()]

                y_in = self.y_inp(torch.cat((y_input.unsqueeze(1), weighted, p_attn), 2))  # (b,1,e+2h+2h)
                self.rnn2_attn.flatten_parameters()
                out, (hid, c) = self.rnn2_attn(y_in.float(),
                                               (hid, c))  # out:(b, 1, h*2), hid:(1, b, h*2)  # out=hid
                out = F.softmax(self.out2(out), 2)  # (b,seq,2h) --> (b,seq,vocab_size)
                return out, hid, c, weighted, p_attn_score, m_attn_score, r_attn_score
