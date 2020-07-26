# coding=utf-8
import torch
import torch.nn as nn
import numpy as np

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class PreStory(nn.Module):
    def __init__(self, args, word_dict, device, encoder, decoder, embedding_dim, pmr_size, hidden_size):
        super(PreStory, self).__init__()
        self.args = args
        self.word_dict = word_dict
        self.device = device
        self.hidden_size = hidden_size
        self.encoder = encoder
        self.decoder = decoder
        self.char_fc = nn.Linear(hidden_size, hidden_size * 2)

    def create_mask(self, src):
        if self.args.gpt2:
            mask = (src != self.word_dict.encode('pad')[0])
            # mask = (src != self.word_dict['pad'])
        else:
            mask = (src != self.word_dict['pad'])
        return mask

    def forward(self, switch, sentence_x, sentence_x_len, context_x, context_x_len, char_idx,
                p_id, m_id, r_id, p_score, m_score, r_score, sentence_y, sentence_y_lenth, epoch, it):
        '''
        sentence_x:   (b, max_len)
        char_embed:   (b, char_len)
        pmr_char_rep: (b, char_num, pmr_size, e)
        '''

        batch_size = sentence_y.size()[0]

        # encoder
        # (b, max_len, 2h)
        x_sen, sen_hid, sen_c, x_con, p_matrix, selfattn_out_sen, selfattn_out_con = self.encoder(sentence_x, sentence_x_len,
                                                                                     context_x, context_x_len,
                                                                                     char_idx, p_id, m_id, r_id,
                                                                                     p_score, m_score, r_score)
        # get initial input of decoder
        if self.args.gpt2:
            decoder_hid = sen_hid  # (1,b,2h)
            decoder_c = torch.randn_like(torch.zeros(1, batch_size, self.hidden_size * 2)).to(self.device)
        elif self.args.context:
            decoder_hid = sen_hid  # (1,b,2h)
            decoder_c = sen_c
        else:
            decoder_hid = sen_hid  # (1,b,2h)
            decoder_c = sen_c

        if self.args.encoder_merge and self.args.context:  # 加入context  ##### <------
            encoded_x = x_sen  # (b, max_len, 2h)
            sentence_x = sentence_x[sentence_x != self.word_dict['bos']].view(
                sentence_x.size(0), -1)
            context_x = context_x[context_x != self.word_dict['eos']].view(
                context_x.size(0), -1)
            encoded_x_idx = torch.cat((context_x, sentence_x), 1)  # (b, max_len)  #
            encoded_con = None
            encoded_con_idx = None
        elif self.args.context:
            encoded_x = x_sen
            encoded_x_idx = sentence_x
            encoded_con = x_con
            encoded_con_idx = context_x

        attn = None
        if self.args.encoder_merge and self.args.self_attention:
            mask_x = self.create_mask(encoded_x_idx)
            mask_c = None
        else:
            mask_x = self.create_mask(encoded_x_idx)
            mask_c = self.create_mask(encoded_con_idx)

        # out_list to store outputs
        decoder_in = sentence_y[:, 0]  # 取每个batch的第一个单词
        for j in range(sentence_y.size(1)):  # max_len 遍历句子的长度，诸逐个时刻输出
            # 1st state
            if j == 0:
                decoder_in = sentence_y[:, j]  # 第一个时刻的单词y_0
                decoder_out, decoder_hid, decoder_c, attn, p_attn_score, m_attn_score, r_attn_score = self.decoder(
                    decoder_in, encoded_x, encoded_x_idx, encoded_con, encoded_con_idx, p_matrix, selfattn_out_sen,
                    selfattn_out_con, decoder_hid, decoder_c, mask_x, mask_c, order=j)
                decoder_in = sentence_y[:, j + 1]  # y_1
                continue
            # remaining states
            else:
                decoder_temp_out, decoder_hid, decoder_c, temp_attn, temp_p_attn_score, temp_m_attn_score, temp_r_attn_score = self.decoder(
                    decoder_in, encoded_x, encoded_x_idx, encoded_con, encoded_con_idx, p_matrix, selfattn_out_sen,
                    selfattn_out_con, decoder_hid, decoder_c, mask_x, mask_c, order=j)
                decoder_out = torch.cat([decoder_out, decoder_temp_out], dim=1)

            # Teacher forcing to select next input
            if self.args.teacher_force:
                decoder_in = decoder_out[:, -1].max(1)[1].squeeze()  # train with sequence outputs
            else:
                if j < sentence_y.size(1) - 1:
                    decoder_in = sentence_y[:, j + 1]  # train with ground truth y_t

        return decoder_out, attn, p_attn_score, m_attn_score, r_attn_score

    def translate(self, args, sentence_x, sentence_x_len, context_x, context_x_len, char_idx, p_id, m_id,
                  r_id, p_score, m_score, r_score, max_length=20):
        batch_size = sentence_x.size()[0]
        # encoder
        x_sen, sen_hid, sen_c, x_con, p_matrix, selfattn_out_sen, selfattn_out_con = self.encoder(sentence_x,
                                                                                                  sentence_x_len,
                                                                                                  context_x,
                                                                                                  context_x_len,
                                                                                                  char_idx, p_id, m_id,
                                                                                                  r_id,
                                                                                                  p_score, m_score,
                                                                                                  r_score)
        if self.args.context:
            decoder_hid = sen_hid  # (1,b,2h)
            decoder_c = sen_c
        else:
            decoder_hid = sen_hid  # (1,b,2h)
            decoder_c = sen_c

        if self.args.encoder_merge and self.args.context:  # 加入context  ##### <------
            encoded_x = x_sen  # (b, max_len, 2h)
            sentence_x = sentence_x[sentence_x != self.word_dict['bos']].view(
                sentence_x.size(0), -1)
            context_x = context_x[context_x != self.word_dict['eos']].view(
                context_x.size(0), -1)
            encoded_x_idx = torch.cat((context_x, sentence_x), 1)  # (b, max_len)  #
            encoded_con = None
            encoded_con_idx = None
        elif self.args.context:
            encoded_x = x_sen
            encoded_x_idx = sentence_x
            encoded_con = x_con
            encoded_con_idx = context_x

        if self.args.encoder_merge and self.args.self_attention:
            mask_x = self.create_mask(encoded_x_idx)
            mask_c = None
        else:
            mask_x = self.create_mask(encoded_x_idx)
            mask_c = self.create_mask(encoded_con_idx)

        # decoder
        # get initial input of decoder
        decoder_in = torch.LongTensor(np.ones(batch_size, dtype=int)).to(self.device) * self.word_dict['bos']
        preds = []
        attns, p_attns, m_attns, r_attns = None, None, None, None
        for i in range(max_length):
            output, decoder_hid, decoder_c, attn, p_attn_score, m_attn_score, r_attn_score = self.decoder(
                decoder_in, encoded_x, encoded_x_idx, encoded_con, encoded_con_idx, p_matrix, selfattn_out_sen,
                selfattn_out_con, decoder_hid, decoder_c, mask_x, mask_c, order=i)
            y = output.max(2)[1].view(batch_size, 1)[0]  # output (batch_size, input_len, vocab_size) (1, 1, 400000)
            preds.append(y.unsqueeze(0))  # y tensor([[2]])
            if self.args.baseline == False:
                if i == 0:
                    p_attns = p_attn_score.unsqueeze(1)
                if i > 0:
                    p_attns = torch.cat([p_attns, p_attn_score.unsqueeze(1)], 1)
            decoder_in = y
        return torch.cat(preds, 1), attns, p_attns, m_attns, r_attns
