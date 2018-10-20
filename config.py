import argparse
import tensorflow as tf
from tensorflow.contrib import slim
import os

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


"""checking arguments"""
def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)


    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args

def parse_args():
    desc = "Tensorflow implementation of OCR"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    parser.add_argument('--dataset', type=str, default='MLT', help='dataset_name')
    parser.add_argument('--epoch', type=int, default=10000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=10, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=500, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=bool, default=False, help='The learning rate decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=10, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0002, help='The learning rate')
    parser.add_argument('--back_bone_type', type=str, default='vgg',
                        help='[vgg,resnet,densenet,inception]')

    parser.add_argument('--classes', type=str, default='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', help='dict_class')
    parser.add_argument('--img_height', type=int, default=64, help='The size of image')
    parser.add_argument('--img_width', type=int, default=300, help='The size of image')
    parser.add_argument('--max_len_word', type=int, default=20, help='The max length of the words')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=bool, default=False, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the attention map')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')

    return check_args(parser.parse_args())
args = parse_args()

