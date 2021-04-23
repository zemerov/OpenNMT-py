#!/usr/bin/env python
from onmt.bin.train import main
import wandb

wandb.init(project="variational-translation")

if __name__ == "__main__":
    main()
