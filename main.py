import arguments
from mixseq import mix_seq


if __name__ == '__main__':
    parser = arguments.parse_args()
    args = parser.parse_args()
    mix_seq(args)