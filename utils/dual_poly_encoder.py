import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPolyEncoder(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.codes_number = 16
        self.hidden_size = 768
        self.poly_codes_for_context = nn.Embedding(self.codes_number,
                                                   self.hidden_size)
        self.poly_codes_for_candidate = nn.Embedding(self.codes_number,
                                                     self.hidden_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.poly_codes_for_context.weight.data.uniform_(-initrange, initrange)
        self.poly_codes_for_candidate.weight.data.uniform_(
            -initrange, initrange)

    def dot_attention(self, q, k, v):
        attention_weights = torch.bmm(q, k.transpose(1, 2))
        scaled_attention_weights = F.softmax(attention_weights, dim=0)
        output = torch.bmm(scaled_attention_weights, v)
        return output

    def mean_pooling(self, last_hidden, attention_mask):
        pass

    def forward(self,
                context_input_ids,
                context_attention_mask,
                candidate_input_ids,
                candidate_attention_mask,
                labels=None):
        batch_size, _ = context_input_ids.shape
        device = context_input_ids.device

        context_output = self.bert(context_input_ids, context_attention_mask)
        context_hidden_state = context_output[0]
        context_cls = context_output[1][:, None, :]
        candidate_output = self.bert(candidate_input_ids, candidate_attention_mask)
        candidate_hidden_state = candidate_output[0]
        candidate_cls = candidate_output[1][:, None, :]
        poly_code_ids = torch.arange(self.codes_number, dtype=torch.int).unsqueeze(0).expand(batch_size, self.codes_number).to(device)

        poly_codes_for_context = self.poly_codes_for_context(poly_code_ids)
        # poly_codes_for_context.shape -> [16, 16, 768]
        # context_hidden_state.shape -> [16, 500, 768]
        # candidate_cls.shape -> [16, 1, 768]

        ctxt_poly_embed = self.dot_attention(poly_codes_for_context, context_hidden_state, context_hidden_state)
        ctxt_embed = self.dot_attention(candidate_cls, ctxt_poly_embed, ctxt_poly_embed)
        negative_score = torch.bmm(candidate_cls, ctxt_embed.transpose(1, 2)).flatten()

        poly_codes_for_candidate = self.poly_codes_for_candidate(poly_code_ids)
        cand_poly_embed = self.dot_attention(poly_codes_for_candidate, candidate_hidden_state, candidate_hidden_state)
        cand_embed = self.dot_attention(context_cls, cand_poly_embed, cand_poly_embed)
        positive_score = torch.bmm(context_cls, cand_embed.transpose(1, 2)).flatten()

        final_score = positive_score - negative_score

        if labels is not None:
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(final_score, labels)
            return loss
        else:
            return final_score
