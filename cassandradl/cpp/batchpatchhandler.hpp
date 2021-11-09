// Copyright 2021 CRS4
// 
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef BATCHPATCHHANDLER_H
#define BATCHPATCHHANDLER_H

#include <cassandra.h>
#include <string>
#include <vector>
#include <future>
#include <utility>
#include <mutex>
#include <opencv2/core.hpp>
#include <eddl/tensor/tensor.h>
#include <ecvl/core.h>
#include <ecvl/augmentations.h>
#include "credentials.hpp"
using namespace std;

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "ThreadPool.hpp"


class BatchPatchHandler{
private:
  // parameters
  bool first_read = true;
  bool connected = false;
  vector<string> keys0; // first batch
  int wb0 = -1; // first buffer
  int thread_par; // thread parallelism
  int num_classes;
  ecvl::Augmentation* aug = NULL;
  string table;
  string label_col;
  string data_col;
  string id_col;
  string username;
  string password;
  vector<string> cassandra_ips;
  string s_cassandra_ips;
  int port = 9042;
  // Cassandra connection and execution
  CassCluster* cluster = cass_cluster_new();
  CassSession* session = cass_session_new();
  const CassPrepared* prepared;
  // concurrency
  ThreadPool* pool;
  mutex mtx;
  // batch parameters
  int chan;
  int height = -1;
  int width;
  int tot_dims;
  // current batch
  const int max_buf = 2;
  vector<int> bs;
  vector<future<pair<shared_ptr<Tensor>, shared_ptr<Tensor>>>> batch;
  vector<shared_ptr<Tensor>> t_feats;
  vector<shared_ptr<Tensor>> t_labs;
  queue<int> read_buf;
  queue<int> write_buf;
  // methods
  void connect();
  void check_connection();
  vector<char> file2buf(string filename);
  ecvl::Image buf2img(const vector<char>& buf);
  void get_img(const CassResult* result, int off, int wb);
  void get_images(const vector<string>& keys, int wb);
  vector<CassFuture*> keys2futures(const vector<string>& keys);
  void future2img(CassFuture* query_future, int off, int wb);
  void setImPar(ecvl::Image* im);
  void allocTens(int wb);
public:
  BatchPatchHandler(int num_classes, ecvl::Augmentation* aug, string table,
		    string label_col, string data_col, string id_col,
		    string username, string cass_pass,
		    vector<string> cassandra_ips, int thread_par=32, int port=9042);
  ~BatchPatchHandler();
  void schedule_batch(const vector<py::object>& keys);
  pair<shared_ptr<Tensor>, shared_ptr<Tensor>> load_batch(const vector<string>& keys, int wb);
  pair<shared_ptr<Tensor>, shared_ptr<Tensor>> block_get_batch();
  void ignore_batch();
};

#endif
