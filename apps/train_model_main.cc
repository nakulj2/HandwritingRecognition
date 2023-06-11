#include <core/model.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// TODO: You may want to change main's signature to take in argc and argv
int main() {
  // TODO: Replace this with code that reads the training data, trains a model,
  // and saves the trained model to a file.

  naivebayes::Model model_training;
  std::string path = "../data/validation_file.txt";
  model_training.TrainModel(path);

  naivebayes::Model model_1;
  std::string path_1 = "../data/training_data.txt";
  model_1.TrainModel(path_1);

  std::ofstream output_file("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/saved_data.txt");
  if (output_file.is_open()) {
    output_file << model_1;
    output_file.close();
  }

  naivebayes::Model model_2;
  std::ifstream inputfile("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/saved_data.txt");
  if (inputfile.is_open()) {
    inputfile >> model_2;
    inputfile.close();
  }

  return 0;
}
