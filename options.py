import argparse
import os


def parse_common_args(parser):
    parser.add_argument('--save_path', type=str, default='ckpt/cnn-inds.ckpt', help='some comment for model or test result dir')
    parser.add_argument('--load_model_path', type=str, default='ckpt/cnn.ckpt',
                        help='model path for pretrain or test')
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--batch_size', type=int, default=64)
    
    """ file path"""
    parser.add_argument('--train_file_path', type=str, default='data/indictors11_train.pkl')
    parser.add_argument('--dev_file_path', type=str, default='data/indictors11_test.pkl')
    parser.add_argument('--test_file_path', type=str, default='data/indictors11_test.pkl')
    parser.add_argument('--vocab_path', type=str, default='data/vacab.pkl')
    " data attribute "
    parser.add_argument('--class_num', type=int, default=3)
    parser.add_argument('--embed_dim', type=int, default=100)
    return parser

def parse_train_args(parser):
    parser = parse_common_args(parser)
    " model name "
    parser.add_argument('--model_name', type=str, default="mcbilstm")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay')
    parser.add_argument('--epoches', type=int, default=20)
    parser.add_argument('--log_intervel', type=int, default=10, help="Number of top K prediction classes (default: 5)")
    parser.add_argument('--eval_interval', type=int, default=10, help="Number of top K prediction classes (default: 5)")
    return parser
def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args



def get_train_model_dir(args):
    model_dir = os.path.join('checkpoints', args.model_type + '_' + args.save_prefix)
    if not os.path.exists(model_dir):
        os.system('mkdir -p ' + model_dir)
    args.model_dir = model_dir

def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fd:
        fd.write(str(args).replace(', ', ',\n'))
def parse_test_args(parser):
    parser = parse_common_args(parser)
    parser.add_argument('--save_viz', action='store_true', help='save viz result in eval or not')
    parser.add_argument('--result_dir', type=str, default='', help='leave blank, auto generated')
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="Run LA_HCN.")

    # hyper-para for datasets
    parser.add_argument('--dataname', type=str, default='enron_2', help="training data.")
    parser.add_argument('--training_data_file', type=str, default='data/large/train/train_almg.json', help="path to training data.")
    parser.add_argument('--validation_data_file', type=str, default='data/large/val/val_almg.json', help="path to validation data.")
    parser.add_argument('--num_classes_list', type=str, default="8,129", help="Number of labels list (depends on the task)")
    parser.add_argument('--glove_file', type=str, default="data/glove6b100dtxt/glove.6B.100d.txt", help="glove embeding file")
    parser.add_argument('--train_or_restore', type=str, default='Train', help="Train or Restore. (default: Train)")

    # hyper-para for training
    parser.add_argument('--BiLSTM', type=bool, default=False, help="True for wipo/BGC; False for Enron/Reuters.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate.")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch Size (default: 256)")
    parser.add_argument('--num_epochs', type=int, default=20, help="Number of training epochs (default: 100)")
    parser.add_argument('--pad_seq_len', type=int, default=300, help="Recommended padding Sequence length of data (depends on the data)")
    parser.add_argument('--embedding_dim', type=int, default=300,help="Dimensionality of character embedding (default: 128)")
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help="Hidden size for bi-lstm layer(default: 256)")
    parser.add_argument('--attention_unit_size', type=int, default=200,
                        help="Attention unit size(default: 200)")
    parser.add_argument('--fc_hidden_size', type=int, default=512,
                        help="Hidden size for fully connected layer (default: 512)")
    parser.add_argument('--dropout', type=float, default=0.5, help= "Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default= 0.0, help="L2 regularization lambda (default: 0.0)")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight of global scores in scores cal")
    parser.add_argument('--norm_ratio', type=float, default=2, help="The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
    parser.add_argument('--decay_steps', type=int, default=5000,
                        help="The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
    parser.add_argument('--decay_rate', type=float, default=0.95, help="Rate of decay for learning rate. (default: 0.95)")
    parser.add_argument('--checkpoint_every', type=int, default=100, help="Save model after this many steps (default: 100)")
    parser.add_argument('--num_checkpoints', type=int, default=5, help="Number of checkpoints to store (default: 5)")

    # hyper-para for prediction
    parser.add_argument('--evaluate_every', type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument('--top_num', type=int, default=5, help="Number of top K prediction classes (default: 5)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for prediction classes (default: 0.5)")


     #日记文件名
    parser.add_argument('--log_file',type=str,default='test.log',help="log result file")
    parser.set_defaults(directed=False)
    return parser.parse_args()
if __name__ == '__main__':
    train_args = get_train_args()
