# Copyright 2021-2 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import io
import numpy as np
import random
import pickle
import PIL.Image
import pyecvl.ecvl as ecvl
import pyeddl.eddl as eddl
from pyeddl.tensor import Tensor
import time
import threading
from tqdm import trange, tqdm
from BPH import BatchPatchHandler
from collections import defaultdict
from cassandradl._cassandra_list_manager import CassandraListManager

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile


class CassandraDataset:
    def __init__(self, auth_prov, cassandra_ips, port=9042):
        """Create ECVL Dataset from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param port: TCP port to connect to (default: 9042)
        :returns:
        :rtype:

        """
        # cassandra parameters
        self.cassandra_ips = cassandra_ips
        self.auth_prov = auth_prov
        self.port = port
        # query variables
        self.table = None
        self.label_col = "label"
        self.label_map = []
        self.id_col = None
        self.data_col = "data"
        self.num_classes = 2
        self.prep = None
        # internal parameters
        self.row_keys = None
        self.augs = []
        self.batch_size = 1
        self.current_split = 0
        self.current_index = []
        self.previous_index = []
        self.batch_handler = []
        self.num_batches = []
        """Number of batches for each split"""
        self._full_batches = False
        self._loaded_batches = []
        self.locks = None
        self.n = None
        self.split = None
        self.num_splits = None
        self._lm_config = None  # configuration of list manager
        self.smooth_eps = 0.0
        """Epsilon value for label smoothing"""
        self.rgb = False
        """Convert images to RGB (default is BGR)"""

    def __del__(self):
        self._ignore_batches()

    def _set_seed(self, seed):
        random.seed(seed)
        if seed is None:
            seed = random.getrandbits(32)
            random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

    def set_config(
        self,
        bs=None,
        table=None,
        num_classes=None,
        id_col=None,
        data_col=None,
        full_batches=None,
        augs=None,
        label_col=None,
        label_map=None,
        rgb=None,
        smooth_eps=None,
        seed=None,
    ):
        """Set data loading configuration parameters

        :param bs: Batch size when loading data
        :param full_batches: Use only full batches
        :param table: Data table, indexed by uuid
        :param num_classes: Number of classes (default: 2)
        :param id_col: Cassandra id column for the images (e.g., 'patch_id')
        :param data_col: Cassandra blob image column (default: 'data')
        :param augs: Data augmentations to be used
        :param label_col: Cassandra label column (e.g., 'label')
        :param label_map: Transformation map for labels (e.g., [1,0] inverts the two classes)
        :param rgb: True if using RGB (otherwise BGR)
        :param smooth_eps: epsilon for label smoothing (e.g., smooth_eps=0.1)
        :param seed: Seed for random generators
        :returns:
        :rtype:

        """
        self._set_seed(seed)
        if augs is None:
            if self.augs == []:  # init augs for the first time
                self.augs = [None] * self.num_splits
        else:
            if len(augs) != self.num_splits:
                raise ValueError(f"Length of augmentations should be {self.num_splits}")
            self.augs = augs
        if bs is not None:
            self.batch_size = bs
        if num_classes is not None:
            self.num_classes = num_classes
        if full_batches is not None:
            self._full_batches = full_batches
        if table is not None:
            self.table = table
        if data_col is not None:
            self.data_col = data_col
        if id_col is not None:
            self.id_col = id_col
        if label_col is not None:
            self.label_col = label_col
        if label_map is not None:
            self.label_map = label_map
        if rgb is not None:
            self.rgb = rgb
        if smooth_eps is not None:
            self.smooth_eps = eps
        self._reset_indexes()

    def save_splits(self, filename):
        """Save list of split ids.

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        stuff = {
            "_lm_config": self._lm_config,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            "table": self.table,
            "label_col": self.label_col,
            "data_col": self.data_col,
            "id_col": self.id_col,
            "row_keys": self.row_keys,
            "split": self.split,
            "seed": self.seed,
        }
        with open(filename, "wb") as f:
            pickle.dump(stuff, f)

    def load_splits(self, filename):
        """Load list of split ids and optionally set batch_size and augmentations.

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        print("Loading splits...")
        with open(filename, "rb") as f:
            stuff = pickle.load(f)

        self._lm_config = stuff["_lm_config"]
        num_classes = stuff["num_classes"]
        label_map = stuff["label_map"]
        table = stuff["table"]
        label_col = stuff["label_col"]
        data_col = stuff["data_col"]
        id_col = stuff["id_col"]
        seed = stuff["seed"]
        self.row_keys = stuff["row_keys"]
        self.split = stuff["split"]

        # reload splits
        self.n = self.row_keys.shape[0]  # set size
        num_splits = len(self.split)
        self._update_split_params(
            num_splits=num_splits,
        )
        # init data table
        self.set_config(
            table=table,
            data_col=data_col,
            num_classes=num_classes,
            id_col=id_col,
            label_col=label_col,
            label_map=label_map,
            seed=seed,
        )

    def _update_split_params(self, num_splits):
        self.num_splits = num_splits
        # create a lock per split
        self.locks = [threading.Lock() for i in range(self.num_splits)]

    def use_splits(self, list_manager):
        """Use splits created by specified list manager

        :param list_manager: List manager to be used
        :returns:
        :rtype:

        """
        self._lm_config = list_manager.get_config()  # save lm configuration
        self.row_keys, self.split = list_manager.get_splits()
        self.n = self.row_keys.shape[0]
        num_splits = len(self.split)
        self._update_split_params(
            num_splits=num_splits,
        )

    def _ignore_batch(self, cs):
        if self._loaded_batches[cs] == 0:
            return  # nothing to wait for
        # wait for (and ignore) batch
        hand = self.batch_handler[cs]
        hand.ignore_batch()

    def _ignore_batches(self):
        # wait for handlers to finish, if running
        if self.batch_handler:
            for cs in range(self.num_splits):
                try:
                    self._ignore_batch(cs)
                except BaseException:
                    pass

    def _reset_indexes(self):
        if not self.num_splits:
            return
        self._ignore_batches()
        self.current_index = []
        self.previous_index = []
        self.batch_handler = []
        self.num_batches = []
        self._loaded_batches = []
        for cs in range(self.num_splits):
            self.current_index.append(0)
            self.previous_index.append(0)
            self._loaded_batches.append(0)
            # set up handlers with augmentations
            aug = self.augs[cs]
            ap = self.auth_prov
            handler = BatchPatchHandler(
                num_classes=self.num_classes,
                label_col=self.label_col,
                data_col=self.data_col,
                id_col=self.id_col,
                label_map=self.label_map,
                table=self.table,
                aug=aug,
                username=ap.username,
                cass_pass=ap.password,
                cassandra_ips=self.cassandra_ips,
                port=self.port,
                smooth_eps=self.smooth_eps,
                rgb=self.rgb,
            )
            self.batch_handler.append(handler)
            if not self._full_batches:
                self.num_batches.append(
                    (self.split[cs].shape[0] + self.batch_size - 1) // self.batch_size
                )
            else:
                self.num_batches.append(self.split[cs].shape[0] // self.batch_size)
            # preload batches
            self._preload_batch(cs)

    def rewind_splits(self, chosen_split=None, shuffle=False):
        """Rewind/reshuffle rows in chosen split and reset its current index

        :param chosen_split: Split to be rewinded. If None rewind all the splits.
        :param shuffle: Apply random permutation (def: False)
        :returns:
        :rtype:

        """
        if chosen_split is None:
            splits = range(self.num_splits)
        else:
            splits = [chosen_split]
        for cs in splits:
            self._ignore_batch(cs)
            with self.locks[cs]:
                if shuffle:
                    self.split[cs] = np.random.permutation(self.split[cs])
                # reset index and preload batch
                self.current_index[cs] = 0
                self._loaded_batches[cs] = 0
                self._preload_batch(cs)

    def mix_splits(self, chosen_splits=[]):
        """Mix data from different splits.

        Note: to be used, e.g., when trainining distributely
        :param chosen_splits: List of chosen splits
        :returns:
        :rtype:

        """
        mix = np.concatenate([self.split[sp] for sp in chosen_splits])
        mix = np.random.permutation(mix)
        start = 0
        for sp in chosen_splits:
            sz = self.split[sp].size
            self.split[sp] = mix[start : start + sz]
            start += sz
            self.rewind_splits(sp)

    def _save_futures(self, rows, cs):
        # choose augmentation
        aug = None
        if len(self.augs) > cs and self.augs[cs] is not None:
            aug = self.augs[cs]
        # get and convert whole batch asynchronously
        handler = self.batch_handler[cs]
        # keys_ = [list(row.values())[0] for row in rows]
        handler.schedule_batch(rows)

    def _compute_batch(self, cs):
        if self._loaded_batches[cs] == 0:
            raise RuntimeError(f"No more batches in split {cs}")
        self._loaded_batches[cs] -= 1  # decrement loaded batches
        hand = self.batch_handler[cs]
        return hand.block_get_batch()

    def set_indexes(self, idx):
        if len(idx) != self.num_splits:
            raise ValueError(f"Length of indexes should be {self.num_splits}")
        self._ignore_batches()
        self.current_index = idx
        for cs in range(self.num_splits):
            self._preload_batch(cs)

    def _preload_batch(self, cs):
        remaining = self.split[cs].shape[0] - self.current_index[cs]
        another_batch = remaining > 0
        if self._full_batches:
            another_batch = remaining >= self.batch_size
        if not another_batch:
            self.previous_index[cs] = self.current_index[cs]  # save old index
            self.current_index[cs] += 1  # register overflow
            return  # end of split, stop prealoding
        idx_ar = self.split[cs][
            self.current_index[cs] : self.current_index[cs] + self.batch_size
        ]
        self.previous_index[cs] = self.current_index[cs]  # save old index
        self.current_index[cs] += idx_ar.size  # increment index
        bb = self.row_keys[idx_ar]
        self._save_futures(bb, cs)
        self._loaded_batches[cs] += 1  # increment loaded batches for this split

    def load_batch(self, split=None):
        """Read a batch from Cassandra DB.

        :param split: Split to read from (default to current_split)
        :returns: (x,y) with x tensor of features and y tensor of labels
        :rtype:

        """
        if split is None:
            cs = self.current_split
        else:
            cs = split
        with self.locks[cs]:
            # start preloading the next batch
            self._preload_batch(cs)
            # compute batch from preloaded raw data
            batch = self._compute_batch(cs)
        return batch

    def load_batch_cross(self, not_splits=[]):
        """Load batch from random split, excluding some (def: [current_split])

        To be used for cross-validation

        :param not_splits: Lists of splits from which data is NOT to be loaded
        :returns:
        :rtype:

        """
        # set splits from which NOT to load
        if not_splits == []:
            ns = [self.current_split]
        else:
            ns = not_splits
        # choose current split among the remaining ones
        ends = np.array([sp.shape[0] for sp in self.split])
        curr = np.array(self.current_index)
        ok = curr <= ends  # valid splits
        if self._full_batches:
            ok = ok * (curr % self.batch_size == 0)
        for sp in ns:  # disable splits in ns
            ok[sp] = False
        sp_list = np.array(range(self.num_splits))
        val_list = sp_list[ok]
        cs = np.random.choice(val_list)
        # return batch from chosen split
        return self.load_batch(cs)
