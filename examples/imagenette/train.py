# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from cassandra.auth import PlainTextAuthProvider
from cassandradl import CassandraDataset, CassandraListManager
from getpass import getpass
from pyeddl.tensor import Tensor
from tqdm import trange, tqdm
import argparse
import models
import numpy as np
import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce


def accuracy(predictions, targets, epsilon=1e-12):
    """
    Computes accuracy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    N = predictions.shape[0]
    ce = np.sum((targets * predictions) + 1e-9) / N
    return ce


def get_net(
    net_name="vgg16",
    in_size=[224, 224],
    num_classes=10,
    lr=1e-5,
    gpus=[1],
    lsb=1,
    init=eddl.HeNormal,
    dropout=None,
    l2_reg=None,
):

    # Network definition
    in_ = eddl.Input([3, in_size[0], in_size[1]])

    out = models.ResNet50(in_, num_classes, init=init, l2_reg=l2_reg, dropout=dropout)

    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(lr),
        ["categorical_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU(gpus, mem="full_mem", lsb=lsb) if gpus else eddl.CS_CPU(),
    )

    eddl.summary(net)

    return net


def main(args):
    num_classes = 10
    size = [224, 224]  # size of images

    # Parse GPU
    if args.gpu:
        gpus = [int(i) for i in args.gpu]
    else:
        gpus = []

    print("GPUs mask: %r" % gpus)
    # Get Network
    net_init = eddl.HeNormal
    net = get_net(
        in_size=size,
        num_classes=num_classes,
        lr=args.lr,
        gpus=gpus,
        lsb=args.lsb,
        init=net_init,
        dropout=args.dropout,
        l2_reg=args.l2_reg,
    )
    out = net.layers[-1]

    ########################################
    ### Set database and read split file ###
    ########################################

    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass("Insert Cassandra user: ")
        cass_pass = getpass("Insert Cassandra password: ")

    # Init Cassandra dataset
    ap = PlainTextAuthProvider(username=cass_user, password=cass_pass)
    id_col = "patch_id"
    label_col = "label"
    clm = CassandraListManager(ap, [cassandra_ip])
    clm.set_config(
        table="imagenette.ids_224",
        grouping_cols=["or_split"],
        id_col=id_col,
        label_col=label_col,
        num_classes=num_classes,
    )
    clm.read_rows_from_db()
    clm.split_setup(
        bags=[[("train",)], [("val",)]],
    )
    cd = CassandraDataset(ap, [cassandra_ip])
    cd.use_splits(clm)
    cd.set_config(
        table="imagenette.data_224",
        bs=args.batch_size,
        id_col=id_col,
        label_col=label_col,
        num_classes=num_classes,
    )

    num_batches_tr = cd.num_batches[0] - 1
    num_batches_val = cd.num_batches[1] - 1

    ################################
    #### Training and evaluation ###
    ################################

    print("Defining metric...", flush=True)

    metric_fn = eddl.getMetric("categorical_accuracy")
    loss_fn = eddl.getLoss("categorical_cross_entropy")

    print("Starting training", flush=True)

    # Main loop across epochs
    for e in range(args.epochs):
        # Training
        cd.current_split = 0  # Set the training split as the current one
        print("Epoch {:d}/{:d} - Training".format(e + 1, args.epochs), flush=True)

        cd.rewind_splits(shuffle=True)
        eddl.reset_loss(net)

        # Looping across batches of training data
        pbar = tqdm(range(num_batches_tr))

        for b_index, b in enumerate(pbar):
            x, y = cd.load_batch()
            x.div_(255.0)
            tx, ty = [x], [y]
            eddl.train_batch(net, tx, ty)

            # print batch train results
            loss = eddl.get_losses(net)[0]
            metr = eddl.get_metrics(net)[0]
            msg = (
                "Epoch {:d}/{:d} (batch {:d}/{:d}) - loss: {:.3f}, acc: {:.3f}".format(
                    e + 1, args.epochs, b + 1, num_batches_tr, loss, metr
                )
            )
            pbar.set_postfix_str(msg)

        pbar.close()

        # Evaluation on validation set batches
        cd.current_split = 1  # Set validation split as the current one

        print("Epoch %d/%d - Evaluation" % (e + 1, args.epochs), flush=True)

        pbar = tqdm(range(num_batches_val))
        tot_loss = []
        tot_acc = []

        for b_index, b in enumerate(pbar):
            x, y = cd.load_batch()
            x.div_(255.0)
            eddl.forward(net, [x])
            output = eddl.getOutput(out)

            result = output.getdata()
            target = y.getdata()
            ca = accuracy(result, target)
            ce = cross_entropy(result, target)

            tot_loss.append(ce)
            tot_acc.append(ca)

            msg = "Epoch {:d}/{:d} (batch {:d}/{:d}) loss: {:.3f}, acc: {:.3f} ".format(
                e + 1,
                args.epochs,
                b + 1,
                num_batches_val,
                np.mean(tot_loss),
                np.mean(tot_acc),
            )
            pbar.set_postfix_str(msg)

        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, metavar="INT", default=10, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, metavar="INT", default=28, help="Batch size"
    )
    parser.add_argument(
        "--lsb",
        type=int,
        metavar="INT",
        default=10,
        help="(Multi-gpu setting) Number of batches to run before synchronizing the weights of the different GPUs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="INT",
        default=None,
        help="Seed of the random generator to manage data load",
    )
    parser.add_argument(
        "--lr", type=float, metavar="FLOAT", default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        metavar="FLOAT",
        default=None,
        help="Float value (0-1) to specify the dropout ratio",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        metavar="FLOAT",
        default=None,
        help="L2 regularization parameter",
    )
    parser.add_argument(
        "--gpu",
        nargs="+",
        default=[1],
        help="Specify GPU mask. For example: 1 to use only gpu0; 1 1 to use gpus 0 and 1; 1 1 1 1 to use gpus 0,1,2,3",
    )
    main(parser.parse_args())
