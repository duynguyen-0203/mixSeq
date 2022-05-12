import argparse


def parse_args():
    r"""Parse all arguments."""
    parser = argparse.ArgumentParser(description='mixSeq model arguments', allow_abbrev=False)

    parser = _add_data_args(parser)
    parser = _add_model_args(parser)

    return parser


def _add_data_args(parser):
    parser.add_argument('--data_name', type=str, default='PhoMT/TedTalks')
    parser.add_argument('--data_path', type=str, default=r'D:\Company\Machine_Translation\Data\TedTalks\train',
                        help='Path to the file containing the training data')
    parser.add_argument('--saved_data_path', type=str, default=r'D:\Company\Machine_Translation\Data\mixSeq',
                        help='Path to save the data after augmentation')

    return parser


def _add_model_args(parser):
    parser.add_argument('--model_name', type=str, default='mixSeq: A Simple Data Augmentation Method')
    parser.add_argument('--seed', type=int, default=36)
    parser.add_argument('--factor_mixseq', type=int, default=2)
    parser.add_argument('--is_contextual', action='store_true', help='Contextual information is available or not')
    parser.add_argument('--sep_token', type=str, default='</s>')

    return parser
