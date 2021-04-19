import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model-policy', type=str, default='resnet')
parser.add_argument('--model-value', type=str, default='resnet')
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--filters', type=int, default=64)
parser.add_argument('--n-blocks', type=int, default=19)
parser.add_argument('--mcts-parallel', type=int, default=32)
parser.add_argument('--travel-count', type=int, default=32)
parser.add_argument('--gibo-path', type=str, default='/home/ubuntu/datasets/gibo3')

args = parser.parse_args()
