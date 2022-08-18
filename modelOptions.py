import argparse
import os
from options import parse_common_args

""" bilistm模型参数"""
def parse_bilstm_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--classes_nums', type=int, default=10)
    parser.add_argument('--lstm_hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--words_dict', type=int, default=4791)
    return parser

def get_bilstm_args():
    parser = argparse.ArgumentParser()
    parser = parse_bilstm_args(parser)
    args = parser.parse_args()
    return args

""" cnn-bilistm-inds模型参数"""
def parse_cnnbilstm_inds_args(parser):
    
    parser = parse_common_args(parser)
    
    parser.add_argument('--word_embedding', type=bool, default=False)
    parser.add_argument('--lstm_hidden_dim', type=int, default=64)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--words_dict', type=int, default=3011)
    parser.add_argument('--n_filters', type=int, default=128)
    parser.add_argument('--kernel_sizes', type=list, default=[3,4,5])
    parser.add_argument('--indexs', type=int, default=9)
    return parser

def get_cnnbilstm_inds_args():
    parser = argparse.ArgumentParser()
    parser = parse_cnnbilstm_inds_args(parser)
    args = parser.parse_args()
    return args

""" cnn模型参数"""
def parse_cnn_args(parser):
    
    parser = parse_common_args(parser)
    parser.add_argument('--lstm_hidden_dim', type=int, default=128)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--words_dict', type=int, default=3011)
    parser.add_argument('--n_filters', type=int, default=256)
    parser.add_argument('--kernel_sizes', type=list, default=[3,4,5])
    parser.add_argument('--indexs', type=int, default=9)
    return parser

def get_cnn_args():
    parser = argparse.ArgumentParser()
    parser = parse_cnn_args(parser)
    args = parser.parse_args()
    return args

""" cnn+indices模型参数"""
def parse_cnn_ind_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--lstm_hidden_dim', type=int, default=128)
    parser.add_argument('--lstm_num_layers', type=int, default=2)
    parser.add_argument('--words_dict', type=int, default=3011)
    parser.add_argument('--n_filters', type=int, default=128)
    parser.add_argument('--kernel_sizes', type=list, default=[3,4,5])
    parser.add_argument('--indexs', type=int, default=9)
    return parser

def get_cnn_ind_args():
    parser = argparse.ArgumentParser()
    parser = parse_cnn_ind_args(parser)
    args = parser.parse_args()
    return args



