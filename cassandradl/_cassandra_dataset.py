# Copyright 2021 CRS4
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

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile


class CassandraListManager:
    def __init__(
        self,
        auth_prov,
        cassandra_ips,
        table,
        id_col,
        label_col,
        grouping_cols,
        label_map=[],
        num_classes=2,
        seed=None,
        port=9042,
    ):
        """Loads the list of patches from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param table: Matadata table with ids
        :param id_col: Cassandra id column for the images (e.g., 'patch_id')
        :param label_col: Cassandra label column (e.g., 'label')
        :param grouping_cols: Columns to group by (e.g., ['patient_id'])
        :param label_map: Transformation map for labels (e.g., [1,0] inverts the two classes)
        :param num_classes: Number of classes (default: 2)
        :param seed: Seed for random generators
        :param port:
        :returns:
        :rtype:

        """
        random.seed(seed)
        np.random.seed(seed)
        # cassandra parameters
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory,
        )
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory,
        )
        profs = {"dict": prof_dict, "tuple": prof_tuple}
        self.cluster = Cluster(
            cassandra_ips,
            execution_profiles=profs,
            protocol_version=4,
            auth_provider=auth_prov,
            port=port,
        )
        self.cluster.connect_timeout = 10  # seconds
        self.sess = self.cluster.connect()
        self.table = table
        # row variables
        self.grouping_cols = grouping_cols
        self.id_col = id_col
        self.label_col = label_col
        self.label_map = label_map
        self.seed = seed
        self.partitions = None
        self.sample_names = None
        self._rows = None
        self.num_classes = num_classes
        self.labs = list(range(self.num_classes))
        # split variables
        self.row_keys = None
        self.max_patches = None
        self.n = None
        self.tot = None
        self._bags = None
        self._cow_rows = None
        self._stats = None
        self._split_stats = None
        self._split_stats_actual = None
        self.actual_split_ratio = None
        self._rows = None
        self.balance = None
        self.split_ratios = None
        self.num_splits = None

    def __del__(self):
        self.cluster.shutdown()

    def read_rows_from_db(self, sample_whitelist=None):
        # get list of all rows
        if self.grouping_cols:
            gc_query = ",".join(self.grouping_cols) + ","
        else:
            gc_query = ""
        query = f"SELECT {gc_query} {self.id_col}, {self.label_col} FROM {self.table} ;"
        res = self.sess.execute(query, execution_profile="tuple")
        all_rows = res.all()
        # sort by grouping keys and labels
        id_idx = len(self.grouping_cols)
        lab_idx = id_idx + 1
        self._rows = defaultdict(lambda: defaultdict(list))
        for row in all_rows:
            gr_val = row[:id_idx]
            id_val = row[id_idx]
            lab_val = row[lab_idx]
            if self.label_map:
                lab_val = self.label_map[lab_val]
            self._rows[gr_val][lab_val].append(id_val)
        self.sample_names = list(self._rows.keys())
        if sample_whitelist is not None:
            self.sample_names = set(sample_whitelist).intersection(self.sample_names)
            self.sample_names = list(self.sample_names)
        random.shuffle(self.sample_names)
        # shuffle all ids in sample bags
        for ks in self._rows.keys():
            for lab in self._rows[ks]:
                random.shuffle(self._rows[ks][lab])
        # save some statistics
        self._after_rows()

    def _after_rows(self):
        # set sample names
        self.sample_names = list(self._rows.keys())
        # set stats
        counters = [[len(s[i]) for i in self.labs] for s in self._rows.values()]
        self._stats = np.array(counters)
        self.tot = self._stats.sum()
        print(f"Read list of {self.tot} patches")

    def set_rows(self, rows):
        self._rows = rows
        self._after_rows()

    def _update_target_params(self, max_patches=None, split_ratios=None, balance=None):
        # update number of patches, default: use all
        self.max_patches = int(self.tot)
        if max_patches is not None:  # use all patches
            self.max_patches = min(max_patches, self.max_patches)
        # update and normalize balance, default: uniform
        self.balance = balance
        if balance is not None:
            self.balance = np.array(balance)
            self.balance = self.balance / self.balance.sum()
        # update and normalize split ratios, default: [1]
        if split_ratios is not None:
            self.split_ratios = np.array(split_ratios)
        if self.split_ratios is None:
            self.split_ratios = np.array([1])
        self.split_ratios = self.split_ratios / self.split_ratios.sum()
        assert self.num_splits == len(self.split_ratios)

    def _split_groups(self):
        """partitioning groups of images in bags

        :returns: nothing. bags are saved in self._bags
        :rtype:

        """
        tots = self._stats.sum(axis=0)
        stop_at = self.split_ratios.reshape((-1, 1)) * tots
        # init bags for splits
        bags = []  # bag-0, bag-1, etc.
        for i in range(self.num_splits):
            bags.append([])
        # no grouping? always use the same bag
        if not self.grouping_cols:
            bags = [[]] * self.num_splits
        # insert patches into bags until they're full
        cows = np.zeros([self.num_splits, self.num_classes]).astype(int)
        curr = random.randint(0, self.num_splits - 1)  # start with random bag
        for (i, p_num) in enumerate(self._stats):
            skipped = 0
            # check if current bag can hold the sample set, if not increment
            # bag
            while (
                (cows[curr] + p_num) > stop_at[curr]
            ).any() and skipped < self.num_splits:
                skipped += 1
                curr += 1
                curr %= self.num_splits
            if skipped == self.num_splits:  # if not found choose a random one
                curr = random.randint(0, self.num_splits - 1)
            bags[curr] += [self.sample_names[i]]
            cows[curr] += p_num
            curr += 1
            curr %= self.num_splits
        # save bags and split statistics
        self._bags = bags
        if not self.grouping_cols:
            self._split_stats = stop_at.round().astype(int)
        else:
            self._split_stats = cows
        self.actual_split_ratio = (
            self._split_stats.sum(axis=1) / self._split_stats.sum()
        )

    def _enough_rows(self, sp, sample_num, lab):
        """Are there other rows available, given bag/sample/label?

        :param sp: split/bag
        :param sample_num: group number
        :param lab: label
        :returns:
        :rtype:

        """
        bag = self._bags[sp]
        sample_name = bag[sample_num]
        num = self._cow_rows[sample_name][lab]
        return num > 0

    def _find_row(self, sp, sample_num, lab):
        """Returns a group/sample which contains a row with a given label

        :param sp: split/bag
        :param sample_num: starting group number
        :param lab: required label
        :returns:
        :rtype:

        """
        max_sample = len(self._bags[sp])
        cur_sample = sample_num
        inc = 0
        while inc < max_sample and not self._enough_rows(sp, cur_sample, lab):
            cur_sample += 1
            cur_sample %= max_sample
            inc += 1
        if inc >= max_sample:  # row not found
            cur_sample = -1
        return cur_sample

    def _fill_splits(self):
        """Insert into the splits, taking into account the target class balance

        :returns:
        :rtype:

        """
        # init counter per each partition
        self._cow_rows = {}
        for sn in self._rows.keys():
            self._cow_rows[sn] = {}
            for l in self._rows[sn].keys():
                self._cow_rows[sn][l] = len(self._rows[sn][l])

        eff_max_patches = self.max_patches * self.split_ratios
        if self.balance is not None:
            bal_max_patches = (self._split_stats / self.balance).min(axis=1)
            eff_max_patches = np.minimum(bal_max_patches, eff_max_patches)
            get_from_class = (
                (eff_max_patches.reshape([-1, 1]) * self.balance).round().astype(int)
            )
        else:
            sample_ratio = self.max_patches / self.tot
            get_from_class = (sample_ratio * self._split_stats).round().astype(int)

        self._split_stats_actual = get_from_class
        tot_patches = get_from_class.sum()
        sp_rows = []
        pbar = tqdm(desc="Choosing patches", total=tot_patches)
        for sp in range(self.num_splits):  # for each split
            sp_rows.append([])
            bag = self._bags[sp]
            max_sample = len(bag)
            for cl in range(self.num_classes):  # fill with each class
                m_class = get_from_class[sp][cl]
                cur_sample = 0
                tot = 0
                while tot < m_class:
                    if not self._enough_rows(sp, cur_sample, self.labs[cl]):
                        cur_sample = self._find_row(sp, cur_sample, self.labs[cl])
                    if cur_sample < 0:  # not found, skip to next class
                        break
                    sample_name = bag[cur_sample]
                    self._cow_rows[sample_name][self.labs[cl]] -= 1
                    idx = self._cow_rows[sample_name][self.labs[cl]]
                    row = self._rows[sample_name][self.labs[cl]][idx]
                    sp_rows[sp].append(row)
                    tot += 1
                    cur_sample += 1
                    cur_sample %= max_sample
                    pbar.update(1)
        pbar.close()
        # build common sample list
        self.split = []
        self.row_keys = []
        start = 0
        for sp in range(self.num_splits):
            self.split.append(None)
            sz = len(sp_rows[sp])
            random.shuffle(sp_rows[sp])
            self.row_keys += sp_rows[sp]
            self.split[sp] = np.arange(start, start + sz)
            start += sz
        self.row_keys = np.array(self.row_keys)
        self.n = self.row_keys.shape[0]  # set size

    def split_setup(
        self,
        max_patches=None,
        split_ratios=None,
        balance=None,
        seed=None,
        bags=None,
    ):
        """(Re)Insert the patches in the splits, according to split and class ratios

        :param max_patches: Number of patches to be read. If None use all patches.
        :param split_ratios: Ratio among training, validation and test. If None use the current value.
        :param balance: Ratio among the different classes. If None use the current value.
        :param seed: Seed for random generators
        :param bags: User provided bags for the each split
        :returns:
        :rtype:

        """
        # seed random generators
        random.seed(seed)
        np.random.seed(seed)
        # if bags are provided, infer split_ratio
        if bags:
            split_ratios = [1] * len(bags)
        # update dataset parameters
        self.num_splits = len(split_ratios)
        self._update_target_params(
            max_patches=max_patches, split_ratios=split_ratios, balance=balance
        )
        # divide groups into bags (saved as self._bags)
        if bags:
            # user provided bags
            self._bags = bags
            self._split_stats = np.zeros([self.num_splits, self.num_classes]).astype(
                int
            )
            # compute stats for bags
            for i in self.labs:
                for e, b in enumerate(self._bags):
                    for f in b:
                        self._split_stats[e][i] += len(self._rows[f][i])
        else:
            # automatic bags
            self._split_groups()

        self.actual_split_ratio = (
            self._split_stats.sum(axis=1) / self._split_stats.sum()
        )
        # fill splits from bags
        self._fill_splits()


# ecvl reader for Cassandra


class CassandraDataset:
    def __init__(self, auth_prov, cassandra_ips, port=9042, seed=None):
        """Create ECVL Dataset from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param port: TCP port to connect to (default: 9042)
        :param seed: Seed for random generators
        :returns:
        :rtype:

        """
        # seed random generators
        random.seed(seed)
        if seed is None:
            seed = random.getrandbits(32)
            random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        print(f"Seed used by random generators: {seed}")
        # cassandra parameters
        self.cassandra_ips = cassandra_ips
        self.auth_prov = auth_prov
        self.port = port
        # query variables
        self.table = None
        self.id_col = None
        self.label_col = None
        self.label_map = []
        self.data_col = "data"
        self.num_classes = None
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
        self._clm = None  # Cassandra list manager
        self.smooth_eps = 0.0
        """Epsilon value for label smoothing"""
        self.rgb = False
        """Convert images to RGB (default is BGR)"""

    def __del__(self):
        self._ignore_batches()

    def init_listmanager(
        self,
        table,
        id_col,
        label_col="label",
        label_map=[],
        grouping_cols=[],
        num_classes=2,
    ):
        """Initialize the Cassandra list manager.

        It takes care of loading/saving the full list of rows from the
        DB and creating the splits according to the user input.

        :param table: Metadata by natural keys
        :param id_col: Cassandra id column for the images (e.g., 'patch_id')
        :param label_col: Cassandra label column (default: 'label')
        :param label_map: Transformation map for labels (e.g., [1,0] inverts the two classes)
        :param grouping_cols: Columns to group by (e.g., ['patient_id'])
        :param num_classes: Number of classes (default: 2)
        :returns:
        :rtype:

        """
        self.id_col = id_col
        self.label_col = label_col
        self.label_map = label_map
        self.num_classes = num_classes
        self._clm = CassandraListManager(
            auth_prov=self.auth_prov,
            cassandra_ips=self.cassandra_ips,
            port=self.port,
            table=table,
            grouping_cols=grouping_cols,
            id_col=self.id_col,
            label_col=self.label_col,
            label_map=self.label_map,
            num_classes=self.num_classes,
            seed=self.seed,
        )

    def set_config(
        self,
        bs=None,
        table=None,
        data_col=None,
        full_batches=None,
        augs=None,
        label_map=None,
        rgb=None,
        smooth_eps=None,
    ):
        """Set data loading configuration parameters

        :param bs: Batch size when loading data
        :param full_batches: Use only full batches
        :param table: Data table, indexed by uuid
        :param data_col: Cassandra blob image column (default: 'data')
        :param augs: Data augmentations to be used
        :param label_map: Transformation map for labels (e.g., [1,0] inverts the two classes)
        :param rgb: True if using RGB (otherwise BGR)
        :param smooth_eps: epsilon for label smoothing (e.g., smooth_eps=0.1)
        :returns:
        :rtype:

        """
        if augs is None and self.augs == []:  # init augs for the first time
            self.augs = [None] * self.num_splits
        else:
            if len(augs) != self.num_splits:
                raise ValueError(f"Length of augmentations should be {self.num_splits}")
            self.augs = augs
        if bs is not None:
            self.batch_size = bs
        if full_batches is not None:
            self._full_batches = full_batches
        if table is not None:
            self.table = table
        if data_col is not None:
            self.data_col = data_col
        if label_map is not None:
            self.label_map = label_map
        if rgb is not None:
            self.rgb = rgb
        if smooth_eps is not None:
            self.smooth_eps = eps
        self._reset_indexes()

    def save_rows(self, filename):
        """Save full list of DB rows to file

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        stuff = (
            self._clm.table,
            self._clm.grouping_cols,
            self.id_col,
            self.num_classes,
            self._clm._rows,
        )

        with open(filename, "wb") as f:
            pickle.dump(stuff, f)

    def load_rows(self, filename):
        """Load full list of DB rows from file

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        print("Loading rows...")
        with open(filename, "rb") as f:
            stuff = pickle.load(f)

        (
            clm_table,
            clm_grouping_cols,
            self.id_col,
            self.num_classes,
            clm_rows,
        ) = stuff

        self.init_listmanager(
            table=clm_table,
            grouping_cols=clm_grouping_cols,
            id_col=self.id_col,
            num_classes=self.num_classes,
        )
        self._clm.set_rows(clm_rows)

    def read_rows_from_db(self, sample_whitelist=None):
        """Read the full list of rows from the DB.

        :param sample_whitelist: Whitelist for group keys
        :returns:
        :rtype:

        """
        self._clm.read_rows_from_db(sample_whitelist)

    def save_splits(self, filename):
        """Save list of split ids.

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        stuff = {
            "clm_table": self._clm.table,
            "clm_grouping_cols": self._clm.grouping_cols,
            "id_col": self.id_col,
            "num_classes": self.num_classes,
            "label_map": self.label_map,
            "table": self.table,
            "label_col": self.label_col,
            "data_col": self.data_col,
            "row_keys": self.row_keys,
            "split": self.split,
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

            clm_table = stuff["clm_table"]
            clm_grouping_cols = stuff["clm_grouping_cols"]
            self.id_col = stuff["id_col"]
            self.num_classes = stuff["num_classes"]
            label_map = stuff["label_map"]
            table = stuff["table"]
            label_col = stuff["label_col"]
            data_col = stuff["data_col"]
            self.row_keys = stuff["row_keys"]
            split = stuff["split"]

        # recreate listmanager
        self.init_listmanager(
            table=clm_table,
            grouping_cols=clm_grouping_cols,
            id_col=self.id_col,
            label_col=label_col,
            label_map=label_map,
            num_classes=self.num_classes,
        )
        # reload splits
        self.split = split
        self.n = self.row_keys.shape[0]  # set size
        num_splits = len(self.split)
        self._update_split_params(
            num_splits=num_splits,
        )
        # init data table
        self.set_config(table=table, data_col=data_col)

    def _update_split_params(self, num_splits):
        self.num_splits = num_splits
        # create a lock per split
        self.locks = [threading.Lock() for i in range(self.num_splits)]

    def split_setup(
        self,
        max_patches=None,
        split_ratios=None,
        balance=None,
        seed=None,
        bags=None,
    ):
        """(Re)Insert the patches in the splits, according to split and class ratios

        :param max_patches: Number of patches to be read. If None use all images.
        :param split_ratios: Ratio among training, validation and test. If None use the current value.
        :param balance: Ratio among the different classes. If None use the current value.
        :param seed: Seed for random generators
        :param bags: User provided bags for the each split
        :returns:
        :rtype:

        """
        self._clm.split_setup(
            max_patches=max_patches,
            split_ratios=split_ratios,
            balance=balance,
            seed=seed,
            bags=bags,
        )
        self.row_keys = self._clm.row_keys
        self.split = self._clm.split
        self.n = self._clm.n
        num_splits = self._clm.num_splits
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
