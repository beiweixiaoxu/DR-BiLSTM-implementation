from drlstm.true_utils import *
from drlstm.true_model import *
from drlstm.true_layers import *
import torch.nn as nn
import torch
import torchsnooper
class singlemodel(DRLSTM_BASE):
    def __init__(self,vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout, num_classes,device,rounds):
        super(singlemodel, self).__init__(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout, num_classes,device)
        self.rounds = rounds
        self.recover_layer1 = nn.Linear(2*self.hidden_size, self.embedding_dim)
        self.recover_layer2 = nn.Linear(2*self.hidden_size, self.hidden_size)
    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)
        # embedding
        embeded_premises = self._embedding(premises)
        embeded_hypotheses = self._embedding(hypotheses)
        _, (final_state_of_hypotheses) = self._encoder1(embeded_hypotheses,
                                                        hypotheses_lengths,
                                                        init_state=None)
        for round in range(self.rounds):
            if round == 0:
                encoded_premises, _ = self._encoder2(embeded_premises,
                                                      premises_lengths,
                                                      init_state=final_state_of_hypotheses)
            else:
                encoded_premises, _ = self._encoder2(encoded_premises,
                                                     premises_lengths,
                                                     init_state=final_state_of_hypotheses)
            if self.rounds != 1 and round != self.rounds - 1:
                encoded_premises = self.recover_layer1(encoded_premises) if self.rounds != 1 else encoded_premises

        _, (final_state_of_premises) = self._encoder1(embeded_premises,
                                                      premises_lengths,
                                                      init_state=None)
        for round in range(self.rounds):
            if round == 0:
                encoded_hypotheses, _ = self._encoder2(embeded_hypotheses,
                                                        hypotheses_lengths,
                                                        init_state=final_state_of_premises)
            else:
                encoded_hypotheses, _ = self._encoder2(encoded_hypotheses,
                                                       hypotheses_lengths,
                                                       init_state=final_state_of_premises)
            if self.rounds != 1 and round != self.rounds - 1:
                encoded_hypotheses = self.recover_layer1(encoded_hypotheses) if self.rounds != 1 else encoded_hypotheses
        # attention
        attended_premises, attended_hypotheses = self._attention(encoded_premises,
                                                                 premises_mask,
                                                                 encoded_hypotheses,
                                                                 hypotheses_mask)
        # enhance
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses - attended_hypotheses,
                                         encoded_hypotheses * attended_hypotheses],
                                        dim=-1)
        # projection
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)
        # infer_encoder
        encoded_hypotheses1, (final_state_of_projected_hypotheses) = self._encoder3(projected_hypotheses,
                                                                  hypotheses_lengths,
                                                                  init_state=None)
        for round in range(self.rounds):
            if round == 0:
                encoded_premises2, _ = self._encoder4(projected_premises,
                                                      premises_lengths,
                                                      init_state=final_state_of_projected_hypotheses)
            else:
                encoded_premises2, _ = self._encoder4(encoded_premises2,
                                                      premises_lengths,
                                                      init_state=final_state_of_projected_hypotheses)
            if self.rounds != 1 and round != self.rounds - 1:
                encoded_premises2 = self.recover_layer2(encoded_premises2) if self.rounds != 1 else encoded_premises2

        encoded_premises1, (final_state_of_projected_premises) = self._encoder3(projected_premises,
                                                                premises_lengths,
                                                                init_state=None)
        for round in range(self.rounds):
            if round == 0:
                encoded_hypotheses2, _ = self._encoder4(projected_hypotheses,
                                                        hypotheses_lengths,
                                                        init_state=final_state_of_projected_premises)
            else:
                encoded_hypotheses2, _ = self._encoder4(encoded_hypotheses2,
                                                        hypotheses_lengths,
                                                        init_state=final_state_of_projected_premises)
            if self.rounds != 1 and round != self.rounds - 1:
                encoded_hypotheses2 = self.recover_layer2(encoded_hypotheses2) if self.rounds != 1 else encoded_hypotheses2
        # pooling
        pooled_premises = self._pooling(encoded_premises1,
                                        encoded_premises2,
                                        premises_mask)
        pooled_hypotheses = self._pooling(encoded_hypotheses1,
                                          encoded_hypotheses2,
                                          hypotheses_mask)
        # cat
        cat_pooled_p_h = torch.cat([pooled_premises, pooled_hypotheses], dim=-1)
        # classification
        logits = self._classification(cat_pooled_p_h)
        # softmax
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits.unsqueeze(1), probabilities



class multimodel(nn.Module):
    def __init__(self,vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.4, # change: dropout=0.5
                 num_classes=3,
                 device="cpu"):
        super(multimodel, self).__init__()
        self.model_round1 = singlemodel(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout, num_classes,device,rounds=1)
        self.model_round2 = singlemodel(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout, num_classes,device,rounds=2)
        self.model_round3 = singlemodel(vocab_size,embedding_dim,hidden_size,embeddings,padding_idx,dropout, num_classes,device,rounds=3)
        self.weight_layer = nn.Linear(3,1)
    def forward(self, premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        logits1, _ = self.model_round1(premises,
                                        premises_lengths,
                                        hypotheses,
                                        hypotheses_lengths)

        logits2, _ = self.model_round2(premises,
                                        premises_lengths,
                                        hypotheses,
                                        hypotheses_lengths)

        logits3, _ = self.model_round3(premises,
                                        premises_lengths,
                                        hypotheses,
                                        hypotheses_lengths)
        logits = torch.cat([logits1, logits2, logits3], dim=1)
        logits = logits.transpose(dim0=2, dim1=1)
        final_logits = self.weight_layer(logits).squeeze(-1)
        final_probabilities = nn.functional.softmax(final_logits, dim=-1)
        return final_logits, final_probabilities
