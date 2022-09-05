import argparse
class hyperparameter():
    def __init__(self):
        parser = argparse.ArgumentParser(description="Parser For Arguments",
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Dataset and Experiment name
        parser.add_argument('-data', dest="dataset", default='FB15K237', help='Dataset to use for the experiment')
        parser.add_argument("-name", default='testrun_', help='Name of the experiment')

        # Training parameters
        parser.add_argument("--gpu", type=str, default='0', help='GPU to use, set -1 for CPU')
        parser.add_argument("--train_strategy", type=str, default='one_to_x', help='Training strategy to use')
        parser.add_argument("--opt", type=str, default='adam', help='Optimizer to use for training')
        parser.add_argument('--neg_num', dest="neg_num", default=1000, type=int,
                            help='Number of negative samples to use for loss calculation')
        parser.add_argument('--batch', dest="batch_size", default=128, type=int, help='Batch size')
        parser.add_argument("--l2", type=float, default=0.0, help='L2 regularization')
        parser.add_argument("--lr", type=float, default=0.001, help='Learning Rate')
        parser.add_argument("--epoch", dest='max_epochs', default=500, type=int, help='Maximum number of epochs')
        parser.add_argument("--num_workers", type=int, default=10, help='Maximum number of workers used in DataLoader')
        parser.add_argument('--seed', dest="seed", default=42, type=int, help='Seed to reproduce results')
        parser.add_argument('--restore', dest="restore", action='store_true',
                            help='Restore from the previously saved model')

        # Model parameters
        parser.add_argument("--lbl_smooth", dest='lbl_smooth', default=0.1, type=float,
                            help='Label smoothing for true labels')
        parser.add_argument("--embed_dim", type=int, default=None,
                            help='Embedding dimension for entity and relation, ignored if k_h and k_w are set')
        parser.add_argument('--bias', dest="bias", action='store_true', help='Whether to use bias in the model')
        parser.add_argument('--form', type=str, default='plain', help='The reshaping form to use')
        parser.add_argument('--k_w', dest="k_w", default=10, type=int, help='Width of the reshaped matrix')
        parser.add_argument('--k_h', dest="k_h", default=20, type=int, help='Height of the reshaped matrix')
        parser.add_argument('--num_filt', dest="num_filt", default=32, type=int,
                            help='Number of filters in convolution')
        parser.add_argument('--ker_sz', dest="ker_sz", default=9, type=int, help='Kernel size to use')
        parser.add_argument('--perm', dest="perm", default=1, type=int, help='Number of Feature rearrangement to use')
        parser.add_argument('--hid_drop', dest="hid_drop", default=0.3, type=float, help='Dropout for Hidden layer')
        parser.add_argument('--feat_drop', dest="feat_drop", default=0.2, type=float, help='Dropout for Feature')
        parser.add_argument('--inp_drop', dest="inp_drop", default=0.2, type=float, help='Dropout for Input layer')

        # Logging parameters
        parser.add_argument('--logdir', dest="log_dir", default='./log/', help='Log directory')
        parser.add_argument('--config', dest="config_dir", default='./config/', help='Config directory')

        self.args = parser.parse_args()

    def get_parse(self):
        return self.args