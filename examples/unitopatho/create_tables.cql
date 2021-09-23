//////////////////////////////////////////////////////////////////////
// unitopatho: https://ieee-dataport.org/open-access/unitopatho

CREATE KEYSPACE unito WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 3};

// Mapping patch metadata to data via uuid
CREATE TABLE unito.ids_7000_224(
  image_id text,
  top_label_name text,
  top_label int,
  type_name text,
  type int,
  grade_name text,
  grade int,
  wsi text,
  roi int,
  mpp float,
  x int,
  y int,
  w int,
  h int,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((wsi, or_split, top_label), roi, x, y, w, h)
);

// Mapping patch uuid to data and label
CREATE TABLE unito.data_7000_224(
  patch_id uuid,
  top_label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE unito.metadata_7000_224(
  image_id text,
  top_label_name text,
  top_label int,
  type_name text,
  type int,
  grade_name text,
  grade int,
  wsi text,
  roi int,
  mpp float,
  x int,
  y int,
  w int,
  h int,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);
// Mapping patch metadata to data via uuid
CREATE TABLE unito.ids_800(
  image_id text,
  top_label_name text,
  top_label int,
  type_name text,
  type int,
  grade_name text,
  grade int,
  wsi text,
  roi int,
  mpp float,
  x int,
  y int,
  w int,
  h int,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((wsi, or_split, top_label), roi, x, y, w, h)
);

// Mapping patch uuid to data and label
CREATE TABLE unito.data_800(
  patch_id uuid,
  top_label int,
  data blob,
  PRIMARY KEY ((patch_id))
);

// Mapping patch uuid to metadata
CREATE TABLE unito.metadata_800(
  image_id text,
  top_label_name text,
  top_label int,
  type_name text,
  type int,
  grade_name text,
  grade int,
  wsi text,
  roi int,
  mpp float,
  x int,
  y int,
  w int,
  h int,
  or_split text,
  patch_id uuid,
  PRIMARY KEY ((patch_id))
);
