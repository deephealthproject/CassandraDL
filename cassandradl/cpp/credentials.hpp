// Copyright 2021 CRS4
// 
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef CREDENTIALS_H
#define CREDENTIALS_H

#include<string>
using namespace std;

class Credentials{
public:
  string password;
  string username;
  Credentials(string p, string u) : password(p), username(u) {}
};

#endif
