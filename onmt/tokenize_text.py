import onmt.transforms.bpe as bpe
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
parser.add_argument('--table', type=str, required=True,
                        help='Path to merge table')
parser.add_argument('--target', type=str, default=None,
                        help='Path to tokenized file')

if __name__ == "__main__":
    opts = parser.parse_args()
    merge_table = bpe.load_subword_nmt_table(opts.table)

    with open(opts.file, encoding='utf-8', mode='r') as file:
        lines = file.readlines()

    target_name = opts.target if opts.target is not None else opts.file + '.tok'

    with open(target_name, encoding='utf-8', mode='w') as file:
        for line in lines:
            tokenized_line, _ = bpe.tokenize_text(merge_table, line.split())
            file.write(' '.join(tokenized_line) + '\n')

