cassandradl.CassandraDataset class
==================================

This class take care of loading batches of images from the DB,
converting the tensors and applying augmentations, if needed.

.. autoapimethod:: cassandradl.CassandraDataset.__init__

The class must be initialized with the credentials and the hostname for
connecting to the Cassandra DB, as in the following example::

  from cassandra_dataset import CassandraDataset
  from cassandra.auth import PlainTextAuthProvider
  
  ## Cassandra connection parameters
  ap = PlainTextAuthProvider(username='user', password='pass')
  cd = CassandraDataset(ap, ['cassandra-db'])


The next step is providing the data loader with the splits (i.e.,
lists of UUIDs identifying the images in the DB), created by a
:class:`cassandradl.ListManager` (e.g., by the
:class:`cassandradl.CassandraListManager`)
       
.. autoapimethod:: cassandradl.CassandraDataset.use_splits
  
After the splits have been read from the `ListManager`, we can
configure the data loader using the ``set_config`` method.

.. autoapimethod:: cassandradl.CassandraDataset.set_config

We can, e.g., specify the table from which the actual data are read
and the batch size::
		   
  cd.set_config(
    table='patches.data_by_uuid',
    bs=32,
    id_col='patch_id',
    label_col="label",
  )

It is also possible to apply ECVL augmentations when loading the
data::

  training_augs = ecvl.SequentialAugmentationContainer(
      [
          ecvl.AugMirror(0.5),
          ecvl.AugFlip(0.5),
          ecvl.AugRotate([-180, 180]),
      ]
  )
  augs = [training_augs, None, None]
  cd.set_config(
    table='patches.data_by_uuid',
    bs=32
    augs=augs,
    id_col='patch_id',
    label_col="label",
  )


To set the batch size and ask the system to only generate full batches
(e.g., 32 images also in the last batch)::

  cd.set_config(
    table='patches.data_by_uuid',
    bs=32,
    full_batches=True,
    id_col='patch_id',
    label_col="label",
  )

After the splits have been created, they can easily be saved (together
with the relevant configuration parameters), using the ``save_splits``
method and then reloaded with ``load_splits``.

.. autoapimethod:: cassandradl.CassandraDataset.save_splits
.. autoapimethod:: cassandradl.CassandraDataset.load_splits

For example::
  
  cd.save_splits(
    'splits/100k_3splits.pckl'
  )

And, to load an already existing split file::
  
  from cassandra_dataset import CassandraDataset
  from cassandra.auth import PlainTextAuthProvider
  
  ## Cassandra connection parameters
  ap = PlainTextAuthProvider(username='user', password='pass')
  cd = CassandraDataset(ap, ['cassandra-db'])
  cd.load_splits(
    'splits/100k_3splits.pckl'
  )
  cd.set_config(bs=32)

.. autoapimethod:: cassandradl.CassandraDataset.load_batch
.. autoapimethod:: cassandradl.CassandraDataset.rewind_splits
.. autoapiattribute:: cassandradl.CassandraDataset.num_batches
  
Once the splits are setup, it is finally possible to load batches of
features and labels and pass them to a DeepHealth application, as
shown in the following example::
  
  epochs = 50
  split = 0 # training
  for _ in range(epochs):
      cd.rewind_splits(shuffle=True)
      for _ in range(cd.num_batches[split]):
          x,y = cd.load_batch(split)
          ## feed features and labels to DL engine [...]
  

