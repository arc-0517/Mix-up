import numpy as np
import sys
from configs import TrainConfig
from dataloaders import make_data_loaders
from torch_trainer.models import build_model
from torch_trainer.trainer import Trainer
import wandb

import warnings
warnings.filterwarnings("ignore")


def main():

    config = TrainConfig.parse_arguments()
    config.save()

    # wandb setting
    if config.wandb:
        wandb.init(project=f"DL_baseline_{config.data_name}", reinit=True)
        wandb.run.name = f'model_{config.model_names}_pretrained_{config.pre_trained}'
        wandb.save()

    # define data loader
    train_loader, valid_loader, test_loader = make_data_loaders(config)

    # define mixup type
    if config.mixup_type == "manifold_mixup":
        config.mixup_hidden = True

    # define model & trainer
    model = build_model(model_name=config.model_name, n_class=config.n_class)

    trainer = Trainer(model=model)
    trainer.compile(ckpt_dir=config.checkpoint_dir,
                    loss_function=config.loss_function,
                    optimizer=config.optimizer,
                    scheduler=config.scheduler,
                    learnig_rate=config.lr_ae,
                    epochs=config.epochs,
                    local_rank=config.local_rank,
                    mixup_type=config.mixup_type,
                    mixup_alpha=config.mixup_alpha,
                    mixup_hidden=config.mixup_hidden)

    best_loss = np.inf
    epoch_start = 1
    results = {'train_loss': [],
               'valid_loss': [],
               'valid_acc': [],
               'test_acc': []}

    # Start Train & Evaluate
    for epoch in range(epoch_start, config.epochs+1):
        # train
        train_loss = trainer.train(epoch, train_loader)
        results['train_loss'].append(train_loss)

        valid_loss, valid_acc = trainer.valid(epoch, valid_loader)
        results['valid_loss'].append(valid_loss)
        results['valid_acc'].append(valid_acc)

        test_acc = trainer.test(epoch, test_loader)
        results['test_acc'].append(test_acc)

        if config.wandb:
            wandb.log({
                "Train Loss": train_loss,
                "Valid Loss": valid_loss,
                "Valid acc": valid_acc,
                "Test acc": test_acc,
            }, step=epoch)

        if valid_loss < best_loss:
            # save results
            trainer.save(epoch, results)
            best_loss = valid_loss

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()