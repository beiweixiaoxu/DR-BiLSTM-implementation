from drlstm.true_layers import *
from drlstm.true_utils import *
import torch
class DRLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        super(DRLSTM, self).__init__()

        self.vocab_size = vocab_size  # 42394
        self.embedding_dim = embedding_dim  # 300
        self.hidden_size = hidden_size  # 450
        self.num_classes = num_classes  # 3
        self.dropout = dropout
        self.device = device

        self.debug = False

        self._embedding = nn.Embedding(self.vocab_size,
                                        self.embedding_dim,
                                        padding_idx=padding_idx,
                                        _weight=embeddings)
        # 300->2d
        self._encoder1 = BilstmEncoder(
                nn.LSTM,
                self.embedding_dim,
                self.hidden_size,
                bidirectional=True
        )
        # 300->2d
        self._encoder2 = BilstmEncoder(
            nn.LSTM,
            self.embedding_dim,
            self.hidden_size,
            bidirectional=True
        )
        # 2d->8d
        self._attention = SoftmaxAttention()

        # 8d->d  maybe two
        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                         self.hidden_size),
                                         nn.ReLU())
        # d->2d
        self._encoder3 = BilstmEncoder(
            nn.LSTM,
            self.hidden_size,
            self.hidden_size,
            bidirectional=True
        )
        # d->2d
        self._encoder4 = BilstmEncoder(
            nn.LSTM,
            self.hidden_size,
            self.hidden_size,
            bidirectional=True
        )

        # 2d->4d(2d -max_pooing-> 2d ; 2d -cat max and avg pooing-> 4d)
        self._pooling = WordSentencePooling()

        # 8d(4d+4d)->d->num_class(3)
        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2 * 4 * self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))
        self.apply(_init_model_weights) # new

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)

        # embedding
        embedded_premises = self._embedding(premises)
        embedded_hypotheses = self._embedding(hypotheses)

        # encoder
        _,(final_state_of_hypotheses) = self._encoder1(embedded_hypotheses,
                                                   hypotheses_lengths,
                                                   init_state = None)
        encoded_premises1,_ = self._encoder1(embedded_premises,
                                             premises_lengths,
                                             init_state=final_state_of_hypotheses)
        _, (final_state_of_premises) = self._encoder2(embedded_premises,
                                                     premises_lengths,
                                                     init_state=None)
        encoded_hypotheses1, _ = self._encoder2(embedded_hypotheses,
                                              hypotheses_lengths,
                                              init_state=final_state_of_premises)
        # attention
        attended_premises, attended_hypotheses = self._attention(encoded_premises1,
                                                                 premises_mask,
                                                                 encoded_hypotheses1,
                                                                 hypotheses_mask)

        # enhance
        enhanced_premises = torch.cat([encoded_premises1,
                                       attended_premises,
                                       encoded_premises1 - attended_premises,
                                       encoded_premises1 * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses1,
                                         attended_hypotheses,
                                         encoded_hypotheses1 - attended_hypotheses,
                                         encoded_hypotheses1 * attended_hypotheses],
                                        dim=-1)
        # projection
        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        # infer_encoder
        _, (final_state_of_projected_hypotheses) = self._encoder3(projected_hypotheses,
                                                        hypotheses_lengths,
                                                        init_state=None)
        encoded_premises2, _ = self._encoder3(projected_premises,
                                              premises_lengths,
                                              init_state=final_state_of_projected_hypotheses)
        _, (final_state_of_projected_premises) = self._encoder4(projected_premises,
                                                      premises_lengths,
                                                      init_state=None)
        encoded_hypotheses2, _ = self._encoder4(projected_hypotheses,
                                                hypotheses_lengths,
                                                init_state=final_state_of_projected_premises)

        # pooling
        pooled_premises = self._pooling(encoded_premises1,
                                        encoded_premises2)
        pooled_hypotheses = self._pooling(encoded_hypotheses1,
                                        encoded_hypotheses2)

        # cat
        cat_pooled_p_h = torch.cat([pooled_premises, pooled_hypotheses],
                                   dim=-1)

        # classification
        logits = self._classification(cat_pooled_p_h)

        # softmax
        probabilities = nn.functional.softmax(logits, dim=-1)
        return logits, probabilities

# new
def _init_model_weights(module):
    """
    Initialise the weights of the model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0