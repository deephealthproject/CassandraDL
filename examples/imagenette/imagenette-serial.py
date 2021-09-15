from PIL import Image
from cassandra.auth import PlainTextAuthProvider
from getpass import getpass
from tqdm import tqdm
import argparse
import io
import numpy as np
import os
import uuid
from cassandra_writer import CassandraWriter

def get_data(path):
    img = Image.open(path).convert('RGB')
    # resize and crop to 160x160
    tg = 160
    sz = np.array(img.size)
    min_d = sz.min()
    sc = float(tg) / min_d
    new_sz = (sc * sz).astype(int)
    img = img.resize(new_sz)
    off = (new_sz.max() - tg) // 2
    if (new_sz[0] > new_sz[1]):
        box = [off, 0, off + tg, tg]
    else:
        box = [0, off, tg, off + tg]
    img = img.crop(box)
    # save to stream
    out_stream = io.BytesIO()
    img.save(out_stream, format='JPEG')
    # write to db
    out_stream.flush()
    data = out_stream.getvalue()
    return(data)


def save_images(cassandra_ip, cass_user, cass_pass):
    auth_prov = PlainTextAuthProvider(cass_user, cass_pass)

    def ret(jobs):
        cw = CassandraWriter(auth_prov, [cassandra_ip],
                             'imagenette.ids_160',
                             'imagenette.data_160',
                             'imagenette.metadata_160',
                             get_data)
        for path, label, or_label, or_split in tqdm(jobs):
            cw.save_image(path, label, or_label, or_split)
    return(ret)


def run(args):
    # Read Cassandra parameters
    try:
        from private_data import cassandra_ip, cass_user, cass_pass
    except ImportError:
        cassandra_ip = getpass("Insert Cassandra's IP address: ")
        cass_user = getpass('Insert Cassandra user: ')
        cass_pass = getpass('Insert Cassandra password: ')

    src_dir = args.src_dir
    jobs = []
    labels = dict()

    ln = 0  # next-label number
    for or_split in ['train', 'val']:
        sp_dir = os.path.join(src_dir, or_split)
        subdirs = [d.name for d in os.scandir(sp_dir) if d.is_dir()]
        for or_label in subdirs:
            # if label is new, assign a new number
            if (or_label not in labels):
                labels[or_label] = ln
                ln += 1
            label = labels[or_label]
            cur_dir = os.path.join(sp_dir, or_label)
            fns = os.listdir(cur_dir)
            for fn in fns:
                path = os.path.join(cur_dir, fn)
                jobs.append((path, label, or_label, or_split))
    save_images(cassandra_ip, cass_user, cass_pass)(jobs)


# parse arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-dir",
        metavar="DIR",
        required=True,
        help="Specifies the input directory for Imagenette")
    run(parser.parse_args())
