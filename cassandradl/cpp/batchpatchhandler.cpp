#include "batchpatchhandler.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/mat.hpp>
#include <ecvl/support_eddl.h>


BatchPatchHandler::~BatchPatchHandler(){
  if (connected){
    cass_session_free(session);
    cass_cluster_free(cluster);
    delete(pool);
  }
}

void BatchPatchHandler::connect(){
  cass_cluster_set_contact_points(cluster, s_cassandra_ips.c_str());
  cass_cluster_set_credentials(cluster, username.c_str(), password.c_str());
  cass_cluster_set_port(cluster, port);
  cass_cluster_set_protocol_version(cluster, CASS_PROTOCOL_VERSION_V4);
  CassFuture* connect_future = cass_session_connect(session, cluster);
  CassError rc = cass_future_error_code(connect_future);
  cass_future_free(connect_future);
  if (rc == CASS_OK) {
    printf("Successfully connected!\n");
  } else {
    throw runtime_error("Error: unable to connect to Cassandra DB. ");
  }
  // assemble query
  stringstream ss;
  ss << "SELECT " << label_col << ", " << data_col <<
    " FROM " << table << " WHERE " << id_col << "=?" << endl;
  string query = ss.str();
  // prepare statement
  CassFuture* prepare_future = cass_session_prepare(session, query.c_str());
  prepared = cass_future_get_prepared(prepare_future);
  if (prepared == NULL) {
    /* Handle error */
    cass_future_free(prepare_future);
    throw runtime_error("Error in query: " + query);
  }
  cass_future_free(prepare_future);
  // init thread pool
  pool = new ThreadPool(thread_par);
}


BatchPatchHandler::BatchPatchHandler(int num_classes, ecvl::Augmentation* aug,
				     string table, string label_col,
				     string data_col, string id_col,
				     string username, string cass_pass,
				     vector<string> cassandra_ips,
				     int thread_par, int port) :
  thread_par(thread_par), num_classes(num_classes), aug(aug),
  table(table), label_col(label_col), data_col(data_col), id_col(id_col),
  username(username), password(cass_pass), cassandra_ips(cassandra_ips),
  port(port)
{
  // init multi-buffering variables
  bs.resize(max_buf);
  batch.resize(max_buf);
  t_feats.resize(max_buf);
  t_labs.resize(max_buf);
  for(int i=0; i<max_buf; ++i)
    write_buf.push(i);
  // join cassandra ip's into comma seperated string
  s_cassandra_ips =
    accumulate(cassandra_ips.begin(), cassandra_ips.end(), string(), 
	       [](const string& a, const string& b) -> string { 
		 return a + (a.length() > 0 ? "," : "") + b; 
	       } );
}

vector<char> BatchPatchHandler::file2buf(string filename){
  ifstream file(filename, ios::binary | ios::ate);
  streamsize size = file.tellg();
  file.seekg(0, ios::beg);
  vector<char> buffer(size);
  if (file.read(buffer.data(), size)){
    file.close();
    return buffer;
  } else
    throw runtime_error("Error reading file");
}

cv::Mat BatchPatchHandler::buf2mat(const vector<char>& buf){
  cv::InputArray ia(buf);
  cv::Mat img = cv::imdecode(ia, cv::IMREAD_UNCHANGED);
  // img is BGR
  return(img);
} 

ecvl::Image BatchPatchHandler::buf2img(const vector<char>& buf){
  cv::Mat m = buf2mat(buf);
  ecvl::Image r = ecvl::MatToImage(m);
  if (aug){
    aug->Apply(r);
  }
  return(r);
}


void BatchPatchHandler::setImPar(ecvl::Image* im){
  height = im->Height();
  width = im->Width();
  chan = im->Channels();
  tot_dims = chan * height * width;
}
void BatchPatchHandler::allocTens(int wb){
  t_feats[wb] = shared_ptr<Tensor>(new Tensor({bs[wb], chan, height, width}));
  t_labs[wb] = shared_ptr<Tensor>(new Tensor({bs[wb], num_classes}));
}

void BatchPatchHandler::get_img(const CassResult* result, int off, int wb){
  // decode result
  const CassRow* row = cass_result_first_row(result);
  if (row == NULL) {
    // Handle error
    throw runtime_error("Error: query returned empty set");
  }
  const CassValue* c_lab =
    cass_row_get_column_by_name(row, label_col.c_str());
  const CassValue* c_data =
    cass_row_get_column_by_name(row, data_col.c_str());
  cass_int32_t lab;
  cass_value_get_int32(c_lab, &lab);
  const cass_byte_t* data;
  size_t sz;
  cass_value_get_bytes(c_data, &data, &sz);
  // convert buffer to image as vector<float>
  vector<char> buf(data, data+sz);
  ecvl::Image im = buf2img(buf);
  if (!im.contiguous_){
        throw runtime_error("Image data not contiguous.");
  }
  // free Cassandra result memory (values included)
  cass_result_free(result);
  ////////////////////////////////////////////////////////////////////////
  // run by just one thread
  ////////////////////////////////////////////////////////////////////////
  mtx.lock();
  // if unset, set images parameters
  if (height<0){
    setImPar(&im);
  }
  // allocate batch if needed
  if (!t_feats[wb]){
    allocTens(wb);
  }
  mtx.unlock();
  ////////////////////////////////////////////////////////////////////////
  // copy image and label to tensors
  float* p_feats =t_feats[wb]->ptr + off*tot_dims;
  uint8_t* p_im = im.data_;
  for(int i=0; i<tot_dims; ++i){
    *(p_feats++) = static_cast<float>(*(p_im++));
  }
  // alternative way: using ecvl ImageToTensor
  // Tensor* tf = t_feats.get();
  // ecvl::ImageToTensor(im, tf, off);

  // convert label 
  float* p_labs = t_labs[wb]->ptr + off*num_classes;
  // int to one-hot
  for(int i=0; i<num_classes; ++i){
    float b = (i==lab)? 1.0 : 0.0;
    *(p_labs++) = b;
  }
}

void BatchPatchHandler::future2img(CassFuture* query_future, int off, int wb){
  // get result (blocking)
  const CassResult* result = cass_future_get_result(query_future);
  if (result == NULL) {
    // Handle error
    const char* error_message;
    size_t error_message_length;
    cass_future_error_message(query_future,
			      &error_message, &error_message_length);
    cass_future_free(query_future);
    throw runtime_error("Error: unable to execute query, " +
			string(error_message));
  }
  cass_future_free(query_future);
  get_img(result, off, wb);
}

vector<CassFuture*> BatchPatchHandler::keys2futures(const vector<string>& keys){
  vector<CassFuture*> futs;
  futs.reserve(keys.size());
  for(auto it=keys.begin(); it!=keys.end(); ++it){
    string id = *it;
    // prepare query
    CassStatement* statement = cass_prepared_bind(prepared);
    CassUuid cuid;
    cass_uuid_from_string(id.c_str(), &cuid);
    cass_statement_bind_uuid_by_name(statement, id_col.c_str(), cuid);
    CassFuture* query_future = cass_session_execute(session, statement);
    cass_statement_free(statement);
    futs.push_back(query_future);
  }
  return(futs);
}


void BatchPatchHandler::get_images(const vector<string>& keys, int wb){
  bs[wb] = keys.size();
  vector<CassFuture*> futs = keys2futures(keys);
  vector<future<void>> asys;
  asys.reserve(bs[wb]);
  for(auto i=0; i!=bs[wb]; ++i){
    // recover data and label
    asys.emplace_back(pool->enqueue(&BatchPatchHandler::future2img, this, futs[i], i, wb));
  }
  // barrier
  for(auto it=asys.begin(); it!=asys.end(); ++it){
    it->wait();
  }
}

pair<shared_ptr<Tensor>, shared_ptr<Tensor>> BatchPatchHandler::load_batch(const vector<string>& keys, int wb){
  // get images and assemble batch
  get_images(keys, wb);
  auto r = make_pair(move(t_feats[wb]), move(t_labs[wb]));
  if (t_feats[wb] || t_labs[wb])
    throw runtime_error("Error: pointers should be null");
  return(r);
}

void BatchPatchHandler::check_connection(){
  if(!connected){
    connect();
    connected = true;
    batch[wb0] = async(launch::async, &BatchPatchHandler::load_batch, this, keys0, wb0);
    read_buf.push(wb0);
  }
}

void BatchPatchHandler::schedule_batch(const vector<py::object>& keys){
  int wb = write_buf.front();
  write_buf.pop();
  // convert uuids to strings
  vector<string> ks;
  ks.reserve(keys.size());
  for(auto it=keys.begin(); it!=keys.end(); ++it){
    string s = py::str(*it);
    ks.push_back(s);
  }
  // if first batch, save it for lazy execution
  if (first_read){
    keys0 = ks;
    wb0 = wb;
    first_read = false;
    return;
  }
  check_connection();
  batch[wb] = async(launch::async, &BatchPatchHandler::load_batch, this, ks, wb);
  read_buf.push(wb);
}

pair<shared_ptr<Tensor>, shared_ptr<Tensor>> BatchPatchHandler::block_get_batch(){
  check_connection();
  // recover
  int rb = read_buf.front();
  read_buf.pop();
  auto b = batch[rb].get();
  auto r = make_pair(shared_ptr<Tensor>(b.first), shared_ptr<Tensor>(b.second));
  write_buf.push(rb);
  return(r);
}

void BatchPatchHandler::ignore_batch(){
  if (!connected){
    first_read = true;
    if(wb0>=0)
      write_buf.push(wb0);
    return;
  }
  // if already connected wait for batch to be computed
  auto b = block_get_batch();
  return;
}
