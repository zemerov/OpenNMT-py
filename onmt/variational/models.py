import numpy as np

from torch import nn
from transformers import BertConfig, BertModel


def solve_sigmoid(desired_proba):
    return np.log(desired_proba / (1 - desired_proba))


class TransformerDropProba(nn.Module):
    """
    Neural network for learning probabilities of dropping merges in bpe merge table
    """

    def __init__(self, merge_table_size, vocab_size, max_seq_len, hidden_size=32, pad_id=0, device='cpu'):
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            pad_token_id=pad_id,
            hidden_size=hidden_size,
            num_hidden_layers=2,  # In 'bert-base-uncased' is 12
            num_attention_heads=2,
            intermediate_size=hidden_size * 4,
            num_labels=merge_table_size,
            max_position_embeddings=max_seq_len,
        )

        self.model = BertModel(config).to(device)

        self.policy_head = nn.Linear(hidden_size, merge_table_size)
        self.sigmoid = nn.Sigmoid()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        """
        :param batch: torch.LongTensor [batch_size, seq_len]
        :return: probabilities: torch.FloatTensor [batch_size, merge_table_size]
        """

        pooled_output = self.model(batch, return_dict=True)['pooler_output']  # [batch_size, hidden_size]

        output = self.sigmoid(self.policy_head(pooled_output))
        value = self.value_head(pooled_output)  # [batch_size, 1]

        return output, value  # Possibly value.squeeze()

    def initialize_weight(self, desired_proba=0.1, logger=None):
        assert 1 > desired_proba > 0, "desired_proba should be in (0, 1)"

        self.policy_head.weight.data.fill_(0)
        self.policy_head.bias.data.fill_(
            solve_sigmoid(1 - desired_proba)
        )  # Model predict dropout mask: 0 - drop merge

        if logger is not None:
            logger.info("Set initial probability distribution to {}".format(desired_proba))
