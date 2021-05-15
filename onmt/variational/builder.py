from onmt.utils.logging import logger
from onmt.variational.models import TransformerDropProba
from onmt.inputters.corpus import ParallelCorpus
from onmt.transforms.tokenize import BpeDropoutTransform

import torch


def build_variational_staff(opts, fields, gpu_rank):
    # Create custom tokenizer, corpus
    logger.info("Creating staff for variational train!")

    variational_transform = BpeDropoutTransform(opts)
    variational_transform.warm_up()

    corpus = ParallelCorpus('', opts.data['iwslt']['path_src'], opts.data['iwslt']['path_tgt'])
    corpus.load_full_text()

    src_pad_idx = fields['src'].base_field.vocab.stoi[fields['src'].base_field.pad_token]
    tgt_pad_idx = fields['tgt'].base_field.vocab.stoi[fields['tgt'].base_field.pad_token]

    if gpu_rank != -1:
        device = torch.device("cuda", gpu_rank)
    else:
        device = torch.device("cpu")

    src_merge_model, src_merge_optimizer = build_bpe_model(
        opts, fields, 'src', variational_transform, device, src_pad_idx
    )

    variational_staff = {
        'src_optim': src_merge_optimizer,
        'tgt_optim': None,
        'tokenizer': variational_transform,
        'src_model': src_merge_model,
        'tgt_model': None,
        'corpus': corpus,
    }  # Fill it with objects

    if not opts.only_src:
        variational_staff['tgt_model'], variational_staff['tgt_optim'] = build_bpe_model(
            opts, fields, 'tgt', variational_transform, device, tgt_pad_idx
        )

    return variational_staff


def build_bpe_model(opts, fields, side, variational_transform, device, pad_idx, max_seq_len=256):
    merge_model = TransformerDropProba(
        merge_table_size=len(variational_transform.tables[side]),
        vocab_size=len(fields[side].base_field.vocab.stoi),
        max_seq_len=max_seq_len,
        pad_id=pad_idx,
        device=device
    )
    merge_model.initialize_weight(opts.tgt_init_proba, logger=logger)
    merge_optimizer = torch.optim.Adam(merge_model.parameters(), lr=opts.variational_lr)

    return merge_model, merge_optimizer
