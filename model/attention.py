import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.attn = nn.Linear(hidden_size * 2 * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size), requires_grad=True)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        energy = energy.permute(0, 2, 1)

        # energy = [batch size, dec hid dim, src len]

        # v = [dec hid dim]

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch size, 1, dec hid dim]

        attention = torch.bmm(v, energy).squeeze(1)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Attention1(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super(Attention1, self).__init__()

        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size

        self.linear_in = nn.Linear(enc_hidden_size * 2, dec_hidden_size, bias=False)
        self.linear_out = nn.Linear(enc_hidden_size * 2 + dec_hidden_size, dec_hidden_size)

    def forward(self, output, context, mask):
        # output: batch_size, output_len, hidden_size
        # context: batch_size, input_len, 2*hidden_size

        batch_size = output.size(0)
        output_len = output.size(1)
        input_len = context.size(1)

        context_in = self.linear_in(context.view(batch_size * input_len, -1)).view(
            batch_size, input_len, -1)  # batch_size, input_len, hidden_size

        # context_in.transpose(1,2): batch_size, hidden_size, input_len
        # output: batch_size, output_len, hidden_size
        attn = torch.bmm(output, context_in.transpose(1, 2))
        # batch_size, output_len, input_len

        attn.data.masked_fill(mask, -1e6)

        attn = F.softmax(attn, dim=2)
        # batch_size, output_len, context_len

        context = torch.bmm(attn, context)
        # batch_size, output_len, hidden_size*2

        output = torch.cat((context, output), dim=2)  # batch_size, output_len, hidden_size*2+hidden_size

        output = output.view(batch_size * output_len, -1)
        output = torch.tanh(self.linear_out(output))
        output = output.view(batch_size, output_len, -1)  # batch_size, output_len, hidden_size
        return output, attn  # output:(batch_size, output_len, hidden_size)，attn:(batch_size, output_len, input_len)


class MatrixAttn(nn.Module):

    def __init__(self, linin, linout):
        super().__init__()
        self.attnlin = nn.Linear(linin, linout)

    def forward(self, dec, emb):
        emb, elen = emb
        emask = torch.arange(0, emb.size(1)).unsqueeze(0).repeat(emb.size(0), 1).long().cuda()
        emask = (emask >= elen.unsqueeze(1)).unsqueeze(1)
        decsmall = self.attnlin(dec)
        unnorm = torch.bmm(decsmall, emb.transpose(1, 2))
        unnorm.masked_fill_(emask, -float('inf'))
        attn = F.softmax(unnorm, dim=2)
        out = torch.bmm(attn, emb)
        return out, attn


class luong_gate_attention(nn.Module):

    def __init__(self, hidden_size, emb_size, prob=0.1):
        super(luong_gate_attention, self).__init__()
        self.hidden_size, self.emb_size = hidden_size, emb_size
        self.linear_enc = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_in = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                       nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.linear_out = nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob),
                                        nn.Linear(hidden_size, hidden_size), nn.SELU(), nn.Dropout(p=prob))
        self.softmax = nn.Softmax(dim=-1)

    def init_context(self, context):
        self.context = context.transpose(0, 1)

    def forward(self, h, selfatt=False):
        if selfatt:
            gamma_enc = self.linear_enc(self.context)  # Batch_size * Length * Hidden_size
            gamma_h = gamma_enc.transpose(1, 2)  # Batch_size * Hidden_size * Length
            weights = torch.bmm(gamma_enc, gamma_h)  # Batch_size * Length * Length
            weights = self.softmax(weights / math.sqrt(512))
            c_t = torch.bmm(weights, gamma_enc)  # Batch_size * Length * Hidden_size
            output = self.linear_out(torch.cat([gamma_enc, c_t], 2)) + self.context
            output = output.transpose(0, 1)  # Length * Batch_size * Hidden_size
        else:
            gamma_h = self.linear_in(h).unsqueeze(2)
            weights = torch.bmm(self.context, gamma_h).squeeze(2)
            weights = self.softmax(weights)
            c_t = torch.bmm(weights.unsqueeze(1), self.context).squeeze(1)
            output = self.linear_out(torch.cat([h, c_t], 1))

        return output, weights
