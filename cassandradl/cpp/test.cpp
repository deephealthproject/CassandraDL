#include<cassandra.h>
#include<iostream>
#include <vector>

#include "batchpatchhandler.hpp"
#include "cass_pass.hpp" // credentials
extern Credentials* cred;

int main(){
  auto h = new BatchPatchHandler
    (10, NULL, "imagenette.data_224", "label", "data", "patch_id",
     {}, cred->username, cred->password, {"127.0.0.1"});
  for(int b=0; b<10; ++b){
    // to test the loading insert some actual UUID from your DB
    vector<string> ks({"011920aa-7201-4ce1-9308-f50599a3def2",
	  "01a1af35-278c-42d3-8314-e9184166ae8d"
	  });
    h->schedule_batch_str(ks);
    auto z = h->block_get_batch();
    cout << b << ":";
    for (auto d: z.first->shape)
      cout << " " << d;
    cout << " --";
    for (auto d: z.second->shape)
      cout << " " << d;
    cout << endl;
  }
}

