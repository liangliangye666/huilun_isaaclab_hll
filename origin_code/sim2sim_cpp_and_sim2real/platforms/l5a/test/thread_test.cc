#include <iostream>
#include <string>
#include <thread>

void PrintHelloWorld(std::string str) { std::cout << str << std::endl; }

int main(int argc, char** argv) {
  std::thread thread_1(PrintHelloWorld, "Hello, thread!");
  thread_1.join();
  return 0;
}