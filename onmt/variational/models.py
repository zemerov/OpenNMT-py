import numpy as np

from torch import nn
from transformers import BertForSequenceClassification, BertConfig


def solve_sigmoid(desired_proba):
    return np.log(desired_proba / (1 - desired_proba))


class LinearDropProba(nn.Module):
    """
    Neural network for learning probabilities of dropping merges in bpe merge table
    """

    def __init__(self, merge_table_size, vocab_size, seq_len, hidden_size=32):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.nonlin = nn.LeakyReLU(0.2)

        self.lin1 = nn.Linear(hidden_size * seq_len, hidden_size * 2)
        self.projection = nn.Linear(hidden_size * 2, merge_table_size)
        self.sigmoid = nn.Sigmoid()

        # Normalization layers
        self.batch_norm1 = nn.BatchNorm1d(num_features=hidden_size * seq_len)
        self.batch_norm2 = nn.BatchNorm1d(num_features=hidden_size * 2)

        self.value_head = nn.Linear(hidden_size * 2, 1)  # Get value for current state

        print("Created Linear model for merge-probabilities")

    def forward(self, batch):
        """
        :param batch: torch.LongTensor [batch_size, seq_len]
        :return: probabilities: torch.FloatTensor [batch_size, merge_table_size]
        """

        batch = self.embedding(batch)  # [batch_size, seq_len, hidden_size]
        batch = self.batch_norm1(
            batch.reshape(batch.shape[0], batch.shape[1] * batch.shape[2])
        )
        batch = self.batch_norm2(
            self.nonlin(self.lin1(batch))
        )  # [batch_size, hidden_size * 2]

        probabilities = self.sigmoid(self.projection(batch))

        value = self.value_head(batch)  # [batch_size, 1]

        return probabilities, value.squeeze()

    def initialize_weight(self, desired_proba=0.1):
        print("Initial probability distribution set to {}".format(desired_proba))
        assert 1 > desired_proba > 0, "desired_proba should be in (0, 1)"

        self.projection.weight.data.fill_(0)  # Set all weights to zero
        self.projection.bias.data.fill_(
            solve_sigmoid(1 - desired_proba)
        )  # Model predict dropout mask: 0 - drop merge


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

        self.model = BertForSequenceClassification(config).to(device)

        self.nonlinear = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.value_head = nn.Linear(merge_table_size, 1)

        print("Created Linear model for merge-probabilities")

    def forward(self, batch):
        """
        :param batch: torch.LongTensor [batch_size, seq_len]
        :return: probabilities: torch.FloatTensor [batch_size, merge_table_size]
        """

        logits = self.model(batch, return_dict=True)['logits']  # [batch_size, merge_table_size]
        value = self.value_head(self.nonlinear(logits))  # [batch_size, 1]

        return self.sigmoid(logits), value  # Possibly value.squeeze()

    def initialize_weight(self, desired_proba=0.1):
        print("Initial probability distribution set to {}".format(desired_proba))
        assert 1 > desired_proba > 0, "desired_proba should be in (0, 1)"
        assert False, "This method is not implemented yet"

        #TODO make it correct
        self.projection.weight.data.fill_(0)  # Set all weights to zero
        self.projection.bias.data.fill_(
            solve_sigmoid(1 - desired_proba)
        )  # Model predict dropout mask: 0 - drop merge
