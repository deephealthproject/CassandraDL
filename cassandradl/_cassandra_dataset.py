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

# pip3 install cassandra-driver
import cassandra
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.policies import TokenAwarePolicy, DCAwareRoundRobinPolicy
from cassandra.cluster import ExecutionProfile

# Handler for large query results


class PagedResultHandler():
    def __init__(self, future):
        self.all_rows = []
        self.error = None
        self.finished_event = threading.Event()
        self.future = future
        self.future.add_callbacks(
            callback=self.handle_page,
            errback=self.handle_error)

    def handle_page(self, rows):
        self.all_rows += rows
        if self.future.has_more_pages:
            self.future.start_fetching_next_page()
        else:
            self.finished_event.set()

    def handle_error(self, exc):
        self.error = exc
        self.finished_event.set()


class CassandraListManager():
    def __init__(self, auth_prov, cassandra_ips, table,
                 partition_cols, id_col, label_col, split_ncols=1,
                 num_classes=2, seed=None, port=9042):
        """Loads the list of patches from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param table: Matadata table with ids
        :param partition_cols: Cassandra partition key (e.g., ['name', 'label'])
        :param id_col: Cassandra id column for the images (e.g., 'patch_id')
        :param label_col: Cassandra label column (e.g., 'label')
        :param split_ncols: How many columns of the partition key are to be considered when splitting data (default: 1)
        :param num_classes: Number of classes (default: 2)
        :param seed: Seed for random generators
        :returns:
        :rtype:

        """
        random.seed(seed)
        np.random.seed(seed)
        # cassandra parameters
        prof_dict = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.dict_factory)
        prof_tuple = ExecutionProfile(
            load_balancing_policy=TokenAwarePolicy(DCAwareRoundRobinPolicy()),
            row_factory=cassandra.query.tuple_factory
        )
        profs = {'dict': prof_dict, 'tuple': prof_tuple}
        self.cluster = Cluster(cassandra_ips,
                               execution_profiles=profs,
                               protocol_version=4,
                               auth_provider=auth_prov,
                               port=port)
        self.cluster.connect_timeout = 10  # seconds
        self.sess = self.cluster.connect()
        self.table = table
        # row variables
        self.partition_cols = partition_cols
        self.split_ncols = split_ncols
        self.id_col = id_col
        self.label_col = label_col
        if (label_col != partition_cols[-1]):
            raise ValueError('Last partition key must be the label: '\
                             +f'{partition_cols[-1]} != {label_col}')
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
        self._rows = None
        self.balance = None
        self.split_ratios = None
        self.num_splits = None

    def __del__(self):
        self.cluster.shutdown()

    def read_rows_from_db(self, scan_par=1, sample_whitelist=None):
        self.partitions = self.sess.execute(
            f"SELECT DISTINCT \
        {', '.join(self.partition_cols)} FROM {self.table} ;",
            execution_profile='tuple',
            timeout=90)
        self.partitions = self.partitions.all()
        if (sample_whitelist is not None):
            parts = self.partitions
            swl = sample_whitelist
            self.partitions = [p for p in parts if p[:self.split_ncols] in swl]
        self.sample_names = {name[:self.split_ncols]
                             for name in self.partitions}
        self.sample_names = list(self.sample_names)
        random.shuffle(self.sample_names)
        query = f"SELECT {self.id_col} FROM {self.table} \
        WHERE {'=? AND '.join(self.partition_cols)}=? ;"
        prep = self.sess.prepare(query)
        self._rows = {}
        for sn in self.sample_names:
            self._rows[sn] = {}
            for l in self.labs:
                self._rows[sn][l] = []
        loc_parts = self.partitions.copy()
        pbar = tqdm(desc='Scanning Cassandra partitions', total=len(loc_parts))
        futures = []
        # while there are partitions to be processed
        while (len(loc_parts) > 0 or len(futures) > 0):
            # fill the pool with samples
            while (len(loc_parts) > 0 and len(futures) < scan_par):
                part = loc_parts.pop()
                l = part[-1]  # label
                sn = part[:self.split_ncols]  # grouping
                res = self.sess.execute_async(prep, part,
                                              execution_profile='dict')
                futures.append((sn, l, PagedResultHandler(res)))
            # check if a query slot can be freed
            for future in futures:
                sn, l, handler = future
                if(handler.finished_event.is_set()):
                    if handler.error:
                        raise handler.error
                    res = handler.all_rows
                    self._rows[sn][l] += res
                    futures.remove(future)
                    pbar.update(1)
                    pbar.set_postfix_str(f'added {len(res):5} patches')
            # sleep 1 ms
            time.sleep(.001)
        pbar.close()
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
        counters = [[len(s[i]) for i in self.labs]
                    for s in self._rows.values()]
        self._stats = np.array(counters)
        self.tot = self._stats.sum()
        print(f'Read list of {self.tot} patches')

    def set_rows(self, rows):
        self._rows = rows
        self._after_rows()

    def _update_target_params(self, max_patches=None,
                              split_ratios=None, balance=None):
        # update number of patches, default: all
        if (max_patches is not None):
            self.max_patches = max_patches
        if (self.max_patches is None):  # if None use all patches
            self.max_patches = int(self.tot)
        # update and normalize split ratios, default: [1]
        if (split_ratios is not None):
            self.split_ratios = np.array(split_ratios)
        if (self.split_ratios is None):
            self.split_ratios = np.array([1])
        self.split_ratios = self.split_ratios / self.split_ratios.sum()
        assert(self.num_splits == len(self.split_ratios))
        # update and normalize balance, default: uniform
        if (balance is not None):
            self.balance = np.array(balance)
        if (self.balance is None):  # default to uniform
            self.balance = np.ones(self.num_classes)
        assert(self.balance.shape[0] == self.num_classes)
        self.balance = self.balance / self.balance.sum()

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
        if (self.split_ncols == 0):
            bags = [[]] * self.num_splits
        # insert patches into bags until they're full
        cows = np.zeros([self.num_splits, self.num_classes])
        curr = random.randint(0, self.num_splits - 1)  # start with random bag
        for (i, p_num) in enumerate(self._stats):
            skipped = 0
            # check if current bag can hold the sample set, if not increment
            # bag
            while (((cows[curr] + p_num) > stop_at[curr]).any() and
                   skipped < self.num_splits):
                skipped += 1
                curr += 1
                curr %= self.num_splits
            if (skipped == self.num_splits):  # if not found choose a random one
                curr = random.randint(0, self.num_splits - 1)
            bags[curr] += [self.sample_names[i]]
            cows[curr] += p_num
            curr += 1
            curr %= self.num_splits
        # save bags
        self._bags = bags

    def _enough_rows(self, sp, sample_num, lab):
        """ Are there other rows available, given bag/sample/label?

        :param sp: split/bag
        :param sample_num: group number
        :param lab: label
        :returns:
        :rtype:

        """
        bag = self._bags[sp]
        sample_name = bag[sample_num]
        num = self._cow_rows[sample_name][lab]
        return (num > 0)

    def _find_row(self, sp, sample_num, lab):
        """ Returns a group/sample which contains a row with a given label

        :param sp: split/bag
        :param sample_num: starting group number
        :param lab: required label
        :returns:
        :rtype:

        """
        max_sample = len(self._bags[sp])
        cur_sample = sample_num
        inc = 0
        while (
            inc < max_sample and not self._enough_rows(
                sp,
                cur_sample,
                lab)):
            cur_sample += 1
            cur_sample %= max_sample
            inc += 1
        if (inc >= max_sample):  # row not found
            cur_sample = -1
        return cur_sample

    def _fill_splits(self, use_all_images=False):
        """ Insert into the splits, taking into account the target class balance

        :returns:
        :rtype:

        """
        # init counter per each partition
        self._cow_rows = {}
        for sn in self._rows.keys():
            self._cow_rows[sn] = {}
            for l in self._rows[sn].keys():
                self._cow_rows[sn][l] = len(self._rows[sn][l])
        borders = self.max_patches * self.balance.cumsum()
        borders = borders.round().astype(int)
        borders = np.pad(borders, [1, 0])
        max_class = [borders[i + 1] - borders[i]
                     for i in range(self.num_classes)]
        max_class = np.array(max_class)
        avail_class = self._stats.sum(axis=0)
        get_from_class = np.min([max_class, avail_class], axis=0)
        tot_patches = get_from_class.sum()
        borders = tot_patches * self.split_ratios.cumsum()
        borders = borders.round().astype(int)
        borders = np.pad(borders, [1, 0])
        max_split = [borders[i + 1] - borders[i]
                     for i in range(self.num_splits)]
        sp_rows = []
        pbar = tqdm(desc='Choosing patches', total=tot_patches)
        for sp in range(self.num_splits):  # for each split
            sp_rows.append([])
            bag = self._bags[sp]
            max_sample = len(bag)
            for cl in range(self.num_classes):  # fill with each class
                tmp = get_from_class[cl] * self.split_ratios.cumsum()
                tmp = tmp.round().astype(int)
                tmp = np.pad(tmp, [1, 0])
                if (use_all_images):
                    m_class = avail_class[cl]
                else:
                    m_class = tmp[sp + 1] - tmp[sp]
                cur_sample = 0
                tot = 0
                while (tot < m_class):
                    if (not self._enough_rows(sp, cur_sample, self.labs[cl])):
                        cur_sample = self._find_row(
                            sp, cur_sample, self.labs[cl])
                    if (cur_sample < 0):  # not found, skip to next class
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
    def split_setup(self, max_patches=None, split_ratios=None,
                    balance=None, seed=None, bags=None,
                    use_all_images=False):
        """(Re)Insert the patches in the splits, according to split and class ratios

        :param max_patches: Number of patches to be read. If None use the current value.
        :param split_ratios: Ratio among training, validation and test. If None use the current value.
        :param balance: Ratio among the different classes. If None use the current value.
        :param seed: Seed for random generators
        :returns:
        :rtype:

        """
        # seed random generators
        random.seed(seed)
        np.random.seed(seed)
        # update dataset parameters
        self.num_splits = len(split_ratios)
        self._update_target_params(max_patches=max_patches,
                                   split_ratios=split_ratios,
                                   balance=balance)
        # divide groups into bags (saved as self._bags)
        if (bags):
            self._bags = bags
        else:
            self._split_groups()
        # fill splits from bags
        self._fill_splits(use_all_images=use_all_images)

# ecvl reader for Cassandra


class CassandraDataset():
    def __init__(self, auth_prov, cassandra_ips, port=9042, seed=None):
        """Create ECVL Dataset from Cassandra DB

        :param auth_prov: Authenticator for Cassandra
        :param cassandra_ips: List of Cassandra ip's
        :param seed: Seed for random generators
        :returns:
        :rtype:

        """
        # seed random generators
        random.seed(seed)
        if (seed is None):
            seed = random.getrandbits(32)
            random.seed(seed)
        np.random.seed(seed)
        self.seed = seed
        print(f'Seed used by random generators: {seed}')
        # cassandra parameters
        self.cassandra_ips = cassandra_ips
        self.auth_prov = auth_prov
        self.port = port
        # query variables
        self.table = None
        self.metatable = None
        self.id_col = None
        self.label_col = None
        self.data_col = None
        self.num_classes = None
        self.prep = None
        # internal parameters
        self.row_keys = None
        self.augs = None
        self.batch_size = None
        self.current_split = 0
        self.current_index = []
        self.previous_index = []
        self.batch_handler = []
        self.num_batches = []
        self.locks = None
        self.n = None
        self.split = None
        self.num_splits = None
        self._clm = None  # Cassandra list manager

    def __del__(self):
        self._ignore_batches()

    def init_listmanager(self, table, partition_cols, id_col,
                         label_col='label', split_ncols=1,
                         num_classes=2, metatable=None):
        """Initialize the Cassandra list manager.

        It takes care of loading/saving the full list of rows from the
        DB and creating the splits according to the user input.

        :param table: Metadata by natural keys
        :param metatable: Metadata by uuid patch_id (optional)
        :param partition_cols: Cassandra partition key (e.g., ['name', 'label'])
        :param id_col: Cassandra id column for the images (e.g., 'patch_id')
        :param label_col: Cassandra label column (default: 'label')
        :param split_ncols: How many columns of the partition key are to be considered when splitting data (default: 1)
        :param num_classes: Number of classes (default: 2)
        :returns:
        :rtype:

        """
        self.id_col = id_col
        self.label_col = label_col
        self.num_classes = num_classes
        self.metatable = metatable
        self._clm = CassandraListManager(auth_prov=self.auth_prov,
                                         cassandra_ips=self.cassandra_ips,
                                         port=self.port,
                                         table=table,
                                         partition_cols=partition_cols,
                                         id_col=self.id_col,
                                         label_col=self.label_col,
                                         split_ncols=split_ncols,
                                         num_classes=self.num_classes,
                                         seed=self.seed)

    def init_datatable(
            self,
            table,
            data_col='data',
            gen_handlers=False):
        """Setup queries for db table containing raw data

        :param table: Data table by ids
        :param data_col: Cassandra blob image column (default: 'data')
        :returns:
        :rtype:

        """
        self.table = table
        self.data_col = data_col
        # if splits are set up, then recreate batch handlers
        if (gen_handlers):
            self._reset_indexes()

    def save_rows(self, filename):
        """Save full list of DB rows to file

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        stuff = (self._clm.table, self._clm.partition_cols,
                 self._clm.split_ncols, self.id_col, self.num_classes,
                 self._clm._rows, self.metatable)

        with open(filename, "wb") as f:
            pickle.dump(stuff, f)

    def load_rows(self, filename):
        """Load full list of DB rows from file

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        print('Loading rows...')
        with open(filename, "rb") as f:
            stuff = pickle.load(f)

        (clm_table, clm_partition_cols, clm_split_ncols, self.id_col,
         self.num_classes, clm_rows, metatable) = stuff

        self.init_listmanager(table=clm_table, metatable=metatable,
                              partition_cols=clm_partition_cols,
                              split_ncols=clm_split_ncols, id_col=self.id_col,
                              num_classes=self.num_classes)
        self._clm.set_rows(clm_rows)

    def read_rows_from_db(self, scan_par=1, sample_whitelist=None):
        """Read the full list of rows from the DB.

        :param scan_par: Increase parallelism while scanning Cassandra partitions. It can lead to DB overloading.
        :returns:
        :rtype:

        """
        self._clm.read_rows_from_db(scan_par, sample_whitelist)

    def save_splits(self, filename):
        """Save list of split ids.

        :param filename: Local filename, as string
        :returns:
        :rtype:

        """
        stuff = (self._clm.table, self._clm.partition_cols,
                 self._clm.split_ncols, self.id_col, self.num_classes,
                 self.table, self.label_col, self.data_col,
                 self.row_keys, self.split, self.metatable)
        with open(filename, "wb") as f:
            pickle.dump(stuff, f)

    def load_splits(self, filename, batch_size=None, augs=None):
        """Load list of split ids and optionally set batch_size and augmentations.

        :param filename: Local filename, as string
        :param batch_size: Dataset batch size
        :param augs: Data augmentations to be used. If None use the current ones.
        :returns:
        :rtype:

        """
        print('Loading splits...')
        with open(filename, "rb") as f:
            stuff = pickle.load(f)

        (clm_table, clm_partition_cols,
         clm_split_ncols, self.id_col, self.num_classes,
         table, label_col, data_col,
         self.row_keys, split, metatable) = stuff

        # recreate listmanager
        self.init_listmanager(table=clm_table, metatable=metatable,
                              partition_cols=clm_partition_cols,
                              split_ncols=clm_split_ncols,
                              id_col=self.id_col, label_col=label_col,
                              num_classes=self.num_classes)
        # init data table
        self.init_datatable(
            table=table,
            data_col=data_col,
            gen_handlers=False)
        # reload splits
        self.split = split
        self.n = self.row_keys.shape[0]  # set size
        num_splits = len(self.split)
        self._update_split_params(num_splits=num_splits, augs=augs,
                                  batch_size=batch_size)
        self._reset_indexes()

    def _update_split_params(self, num_splits, augs=None, batch_size=None):
        # update batch_size, default: 8
        if (batch_size is not None):
            self.batch_size = batch_size
        if (self.batch_size is None):
            self.batch_size = 8
        # update augmentations, default: []
        if (augs is not None):
            self.augs = augs
        if (self.augs is None):
            self.augs = []
        self.num_splits = num_splits
        # create a lock per split
        self.locks = [threading.Lock() for i in range(self.num_splits)]

    def split_setup(self, max_patches=None, split_ratios=None, augs=None,
                    balance=None, batch_size=None, seed=None, bags=None, use_all_images=False):
        """(Re)Insert the patches in the splits, according to split and class ratios

        :param max_patches: Number of patches to be read. If None use the current value.
        :param split_ratios: Ratio among training, validation and test. If None use the current value.
        :param augs: Data augmentations to be used. If None use the current ones.
        :param balance: Ratio among the different classes. If None use the current value.
        :param batch_size: Batch size. If None use the current value.
        :param seed: Seed for random generators
        :returns:
        :rtype:

        """
        self._clm.split_setup(max_patches=max_patches,
                              split_ratios=split_ratios,
                              balance=balance, seed=seed, bags=bags,
                              use_all_images=use_all_images)
        self.row_keys = self._clm.row_keys
        self.split = self._clm.split
        self.n = self._clm.n
        num_splits = self._clm.num_splits
        self._update_split_params(num_splits=num_splits, augs=augs,
                                  batch_size=batch_size)
        self._reset_indexes()

    def _ignore_batch(self, cs):
        if (self.current_index[cs] > self.split[cs].shape[0]):
            return  # end of split, nothing to wait for
        # wait for (and ignore) batch
        hand = self.batch_handler[cs]
        hand.ignore_batch()

    def _ignore_batches(self):
        # wait for handlers to finish, if running
        if (self.batch_handler):
            for cs in range(self.num_splits):
                try:
                    self._ignore_batch(cs)
                except BaseException:
                    pass

    def _reset_indexes(self):
        self._ignore_batches()

        self.current_index = []
        self.previous_index = []
        self.batch_handler = []
        self.num_batches = []
        for cs in range(self.num_splits):
            self.current_index.append(0)
            self.previous_index.append(0)
            # set up handlers with augmentations
            if (len(self.augs) > cs):
                aug = self.augs[cs]
            else:
                aug = None
            ap = self.auth_prov
            handler = BatchPatchHandler(num_classes=self.num_classes,
                                        label_col=self.label_col,
                                        data_col=self.data_col,
                                        id_col=self.id_col,
                                        table=self.table, aug=aug,
                                        username=ap.username,
                                        cass_pass=ap.password,
                                        cassandra_ips=self.cassandra_ips,
                                        port=self.port)
            self.batch_handler.append(handler)
            self.num_batches.append(
                (self.split[cs].shape[0] + self.batch_size - 1) // self.batch_size)
            # preload batches
            self._preload_batch(cs)

    def set_batchsize(self, bs):
        """Change dataset batch size

        :param bs: New batch size
        :returns:
        :rtype:

        """
        self.batch_size = bs
        self._reset_indexes()

    def set_augmentations(self, augs):
        """Change used augmentations

        :param augs: Data augmentations to be used.
        :returns:
        :rtype:

        """
        # update augmentations, default: []
        if (augs is not None):
            self.augs = augs
        if (self.augs is None):
            self.augs = []
        self._reset_indexes()

    def rewind_splits(self, chosen_split=None, shuffle=False):
        """Rewind/reshuffle rows in chosen split and reset its current index

        :param chosen_split: Split to be rewinded. If None rewind all the splits.
        :param shuffle: Apply random permutation (def: False)
        :returns:
        :rtype:

        """
        if (chosen_split is None):
            splits = range(self.num_splits)
        else:
            splits = [chosen_split]
        for cs in splits:
            self._ignore_batch(cs)
            with self.locks[cs]:
                if (shuffle):
                    self.split[cs] = np.random.permutation(self.split[cs])
                # reset index and preload batch
                self.current_index[cs] = 0
                self._preload_batch(cs)

    def mix_splits(self, chosen_splits=[]):
        """ Mix data from different splits.

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
            self.split[sp] = mix[start:start + sz]
            start += sz
            self.rewind_splits(sp)

    def _save_futures(self, rows, cs):
        # choose augmentation
        aug = None
        if (len(self.augs) > cs and self.augs[cs] is not None):
            aug = self.augs[cs]
        # get and convert whole batch asynchronously
        handler = self.batch_handler[cs]
        keys_ = [list(row.values())[0] for row in rows]
        handler.schedule_batch(keys_)

    def _compute_batch(self, cs):
        hand = self.batch_handler[cs]
        return(hand.block_get_batch())

    def set_indexes(self, idx):
        if (len(idx) != self.num_splits):
            raise ValueError(f'Length of indexes should be {self.num_splits}')
        self._ignore_batches()
        self.current_index = idx
        for cs in range(self.num_splits):
            self._preload_batch(cs)

    def _preload_batch(self, cs):
        if (self.current_index[cs] >= self.split[cs].shape[0]):
            self.previous_index[cs] = self.current_index[cs]  # save old index
            self.current_index[cs] += 1  # register overflow
            return  # end of split, stop prealoding
        idx_ar = self.split[cs][self.current_index[cs]:
                                self.current_index[cs] + self.batch_size]
        self.previous_index[cs] = self.current_index[cs]  # save old index
        self.current_index[cs] += idx_ar.size  # increment index
        bb = self.row_keys[idx_ar]
        self._save_futures(bb, cs)

    def load_batch(self, split=None):
        """Read a batch from Cassandra DB.

        :param split: Split to read from (default to current_split)
        :returns: (x,y) with x tensor of features and y tensor of labels
        :rtype:

        """
        if (split is None):
            cs = self.current_split
        else:
            cs = split
        with self.locks[cs]:
            # start preloading the next batch
            self._preload_batch(cs)
            # compute batch from preloaded raw data
            batch = self._compute_batch(cs)
        return(batch)

    def load_batch_cross(self, not_splits=[]):
        """Load batch from random split, excluding some (def: [current_split])

        To be used for cross-validation

        :param not_splits: Lists of splits from which data is NOT to be loaded
        :returns:
        :rtype:

        """
        # set splits from which NOT to load
        if (not_splits == []):
            ns = [self.current_split]
        else:
            ns = not_splits
        # choose current split among the remaining ones
        ends = np.array([sp.shape[0] for sp in self.split])
        curr = np.array(self.current_index)
        ok = curr <= ends  # valid splits
        for sp in ns:  # disable splits in ns
            ok[sp] = False
        sp_list = np.array(range(self.num_splits))
        val_list = sp_list[ok]
        cs = np.random.choice(val_list)
        # return batch from chosen split
        return (self.load_batch(cs))
