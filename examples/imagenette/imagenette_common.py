# Copyright 2021 CRS4
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import tqdm
import io
import numpy as np
import os
import uuid
from cassandradl import CassandraWriter


def get_data(path):
    img = Image.open(path).convert("RGB")
    # resize and crop to 224x224
    tg = 224
    sz = np.array(img.size)
    min_d = sz.min()
    sc = float(tg) / min_d
    new_sz = (sc * sz).astype(int)
    img = img.resize(new_sz)
    off = (new_sz.max() - tg) // 2
    if new_sz[0] > new_sz[1]:
        box = [off, 0, off + tg, tg]
    else:
        box = [0, off, tg, off + tg]
    img = img.crop(box)
    # save to stream
    out_stream = io.BytesIO()
    img.save(out_stream, format="JPEG")
    # write to db
    out_stream.flush()
    data = out_stream.getvalue()
    return data


def save_images(cassandra_ip, cass_user, cass_pass):
    auth_prov = PlainTextAuthProvider(cass_user, cass_pass)

    def ret(jobs):
        cw = CassandraWriter(
            auth_prov,
            [cassandra_ip],
            table_ids="imagenette.ids_224",
            table_data="imagenette.data_224",
            table_metadata="imagenette.metadata_224",
            id_col="patch_id",
            label_col="label",
            data_col="data",
            cols=["or_split", "or_label"],
            get_data=get_data,
        )
        for path, label, partition_items in tqdm(jobs):
            cw.save_image(path, label, partition_items)

    return ret


def get_jobs(src_dir):
    jobs = []
    labels = dict()

    ln = 0  # next-label number
    for or_split in ["train", "val"]:
        sp_dir = os.path.join(src_dir, or_split)
        subdirs = [d.name for d in os.scandir(sp_dir) if d.is_dir()]
        for or_label in subdirs:
            # if label is new, assign a new number
            if or_label not in labels:
                labels[or_label] = ln
                ln += 1
            label = labels[or_label]
            partition_items = (or_split, or_label)
            cur_dir = os.path.join(sp_dir, or_label)
            fns = os.listdir(cur_dir)
            for fn in fns:
                path = os.path.join(cur_dir, fn)
                jobs.append((path, label, partition_items))
    return jobs
