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
