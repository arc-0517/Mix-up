import os
import copy
import json
import argparse
import datetime


class TrainConfig(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.base_parser(),
            cls.data_parser(),
            cls.modeling_parser(),
            cls.mixup_parser()
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def checkpoint_dir(self) -> str:
        # TODO: add if needed
        ckpt = os.path.join(
            self.checkpoint_root,
            f'mixup_type+{self.mixup_type}',
            f'beta_value+{self.mixup_alpha}',
            f'model+{self.model_name}',
            self.hash  # ...
        )

        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def base_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Base", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./save_results')
        parser.add_argument('--random_state', type=int, default=0)
        parser.add_argument('--verbose', type=bool, default=True)
        parser.add_argument('--confusion_matrix', type=bool, default=True)
        parser.add_argument('--wandb', type=bool, default=False)
        parser.add_argument('--wandb_project', type=str, default='')
        parser.add_argument('--wandb_run', type=str, default='')
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data_dir', type=str, default='E:/keewon_code/DL_training_baseline/data')
        parser.add_argument('--data_name', type=str, default='dogcat')
        parser.add_argument('--valid_ratio', type=float, default=0.2)
        parser.add_argument('--shuffle_dataset', type=bool, default=True)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--test_batch_size', type=int, default=128)

        return parser

    @staticmethod
    def modeling_parser() -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser("Modeling", add_help=False)
        parser.add_argument('--model_name', type=str, default='resnet18', choices=['resnet18', 'resnet34'])
        parser.add_argument('--pre_trained', type=bool, default=True, choices=[True, False])
        parser.add_argument('--n_class', type=int, default=2)
        parser.add_argument('--loss_function', type=str, default="ce", choices=["ce", "mse"])
        parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "sgd", "adamW"])
        parser.add_argument('--scheduler', type=str, default="cosine")
        parser.add_argument('--lr_ae', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--local_rank', type=int, default=0)

        return parser

    @staticmethod
    def mixup_parser() -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser("Mixup", add_help=False)
        parser.add_argument('--mixup_type', type=str, default='manifold_mixup', choices=['mix_up', 'cut_mix', 'manifold_mixup', 'no_mixup'])
        parser.add_argument('--mixup_hidden', type=bool, default=False, choices=[True, False])
        parser.add_argument('--mixup_alpha', type=float, default=1.0)

        return parser