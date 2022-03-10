# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from cassandradl import CassandraDataset, CassandraListManager

import pyecvl.ecvl as ecvl
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import trange, tqdm
import numpy as np
from time import sleep

# Read Cassandra parameters
try:
    from private_data import cassandra_ip, cass_user, cass_pass
except ImportError:
    cassandra_ip = getpass("Insert Cassandra's IP address: ")
    cass_user = getpass("Insert Cassandra user: ")
    cass_pass = getpass("Insert Cassandra password: ")

# Init Cassandra dataset
ap = PlainTextAuthProvider(username=cass_user, password=cass_pass)

# Create three splits, with ratio 70, 20, 10 and balanced classes
# group by WSI, so that patches from the same slide don't end up in different splits
id_col = "patch_id"
label_col = "top_label"
num_classes = 6
clm = CassandraListManager(ap, [cassandra_ip])
clm.set_config(
    table="unito.ids_7000_224",
    id_col=id_col,
    label_col=label_col,
    grouping_cols=["wsi"],
    num_classes=num_classes,
)
clm.read_rows_from_db()
clm.split_setup(split_ratios=[7, 2, 1], balance=[1] * 6)
cd = CassandraDataset(ap, [cassandra_ip])
cd.use_splits(clm)
cd.set_config(
    bs=32,
    table="unito.data_7000_224",
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)

for _ in range(5):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[0]):
        x, y = cd.load_batch()

# Create two splits using the original train/test partition
# and loading all the images, ignoring balance
# Read images applying augmentations

training_augs = ecvl.SequentialAugmentationContainer(
    [
        ecvl.AugMirror(0.5),
        ecvl.AugFlip(0.5),
        ecvl.AugRotate([-180, 180]),
        # ecvl.AugAdditivePoissonNoise([0, 10]),
        # ecvl.AugGammaContrast([0.5, 1.5]),
        # ecvl.AugGaussianBlur([0, 0.8]),
        # ecvl.AugCoarseDropout([0, 0.3], [0.02, 0.05], 0.5),
    ]
)
augs = [training_augs, None]

clm = CassandraListManager(ap, [cassandra_ip])
clm.set_config(
    table="unito.ids_7000_224",
    grouping_cols=["or_split"],
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)
clm.read_rows_from_db()
clm.split_setup(
    bags=[[("train",)], [("test",)]],
)
cd = CassandraDataset(ap, [cassandra_ip])
cd.use_splits(clm)
cd.set_config(
    bs=32,
    table="unito.data_7000_224",
    augs=augs,
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)

for _ in range(5):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[0]):
        x, y = cd.load_batch()


# Create two splits using the original train/test partition
# and loading all the images, ignoring balance

clm = CassandraListManager(ap, [cassandra_ip])
clm.set_config(
    table="unito.ids_800",
    grouping_cols=["or_split"],
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)
clm.read_rows_from_db()
clm.split_setup(
    bags=[[("train",)], [("test",)]],
)
cd = CassandraDataset(ap, [cassandra_ip])
cd.use_splits(clm)
cd.set_config(
    bs=8,
    table="unito.data_800",
    id_col=id_col,
    label_col=label_col,
    num_classes=num_classes,
)

# load from test set
cd.current_split = 1
for _ in range(5):
    cd.rewind_splits(shuffle=True)
    for i in trange(cd.num_batches[1]):
        x, y = cd.load_batch()
