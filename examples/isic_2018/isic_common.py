# Copyright 2021-2 CRS4
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
import yaml
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
            table_ids="isic.ids_224",
            table_data="isic.data_224",
            table_metadata="isic.metadata_224",
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
    fn = "isic_classification_2018.yml"
    fn = os.path.join(src_dir, fn)
    print("Reading YAML file...", flush=True)
    with open(fn, "r") as f:
        isic = yaml.safe_load(f)
    print("Resizing and inserting in DB...", flush=True)
    jobs = []
    labels = dict(enumerate(isic["classes"]))
    labels = {v: k for k, v in labels.items()}
    for or_split in isic["split"].keys():
        for num in isic["split"][or_split]:
            item = isic["images"][num]
            or_label = item["label"]
            label = labels[or_label]
            partition_items = (or_split, or_label)
            path = os.path.join(src_dir, item["location"])
            jobs.append((path, label, partition_items))
    return jobs
