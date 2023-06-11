#include <core/model.h>

#include <fstream>
#include <iostream>
#include <vector>

namespace naivebayes {

const void Model::TrainModel(std::string & path) {

  std::ifstream input_file(path);

  size_ = SizeOfImage(path);

  if (input_file.is_open()) {
    std::string label = "";
    std::string picture = "";

    while (getline(input_file, label)) {
      size_t size_check = 1;
      std::string send;

      while (getline(input_file, send)) {
        picture += send;
        picture += '\n';

        if (send.length() != size_) {
          std::cout<<send.size()<< std::endl;
          std::cout<<size_;
          throw std::invalid_argument("Pictures have different sizes");
        }

        if (size_check == size_) {
          break;
        } else {
          size_check++;
        }
      }

      Image image(std::stoi(label), picture);
      map_of_priors_[std::stoi(label)]++;
      images_.emplace_back(image);

      picture = "";
      label = "";
    }

    input_file.close();
  } else {
    throw std::invalid_argument("File does not exist");
  }

  size_of_map_ = map_of_priors_.size();
  ProbabilityMatrices();
}

void Model::ProbabilityMatrices() {

  probability_matrix_ = std::vector<std::vector<std::vector<std::vector<float>>>>
      (size_of_map_,std::vector<std::vector<std::vector<float>>>
          (3, std::vector<std::vector<float>>
              (size_, std::vector<float>(size_))));

  for (Image image : images_) {
    int label = image.GetLabel();
    std::vector<std::vector<int>> pixel_vector = image.GetPixels();

    for (size_t i = 0; i < size_; ++i) {
      for (size_t j = 0; j < size_; ++j) {
        if (pixel_vector[i][j] == 0) {
          probability_matrix_[label][0][i][j] += 1;
        } else if (pixel_vector[i][j] == 1) {
          probability_matrix_[label][1][i][j] += 1;
        } else {
          probability_matrix_[label][2][i][j] += 1;
        }
      }
    }
  }

  LaplaceProbability();
}

void Model::LaplaceProbability() {
  for (size_t n = 0; n < size_; n++) {
    for (size_t m = 0; m < size_; m++) {
      for (size_t j = 0; j < map_of_priors_.size(); j++) {
        for (size_t i = 0; i < 3; i++) {
          probability_matrix_[j][i][m][n] =
              (float)(probability_matrix_[j][i][m][n] + kSmoothingConstant) /
              (size_of_map_ * kSmoothingConstant + map_of_priors_[j]);
        }
      }
    }
  }

  for (int i = 0; i < map_of_priors_.size(); i++) {
    map_of_priors_[i] = (float)(map_of_priors_[i] + kSmoothingConstant) /
                        (size_of_map_ * kSmoothingConstant + images_.size());
  }
}

std::ostream& operator<<(std::ostream& os, const Model& model) {

  if (os.bad()) {
    throw std::invalid_argument("File does not exist");
  }

  os << model.size_ << '\n';
  os << model.map_of_priors_.size() << '\n';

  for (size_t i = 0; i < model.size_; i++) {
    for (size_t j = 0; j < model.size_; j++) {
      for (size_t s = 0; s < model.map_of_priors_.size(); s++) {
        for (size_t l = 0; l < 3; l++) {
          os << model.probability_matrix_[s][l][j][i] << '\n';
        }
      }
    }
  }

  for (size_t i = 0; i < model.map_of_priors_.size(); i++) {
    os << model.map_of_priors_.at(i) << '\n';
  }

  return os;
}

std::istream& operator>>(std::istream& is, Model& model) {

  if (is.bad()) {
    throw std::invalid_argument("File does not exist");
  }

  std::string sizestring;
  std::getline(is, sizestring);
  model.size_ = std::stoi(sizestring);

  std::string mapsize;
  std::getline(is, mapsize);
  model.size_of_map_ = std::stoi(mapsize);

  model.probability_matrix_ = std::vector<std::vector<std::vector<std::vector<float>>>>
      (model.size_of_map_,std::vector<std::vector<std::vector<float>>>
          (3, std::vector<std::vector<float>>
              (model.size_, std::vector<float>(model.size_))));

  for (int i = 0; i < model.size_; i++) {
    for (int j = 0; j < model.size_; j++) {
      for (int g = 0; g < model.size_of_map_; g++) {
        for (int h = 0; h < 3; h++) {
          std::string num;
          std::getline(is, num);
          model.probability_matrix_[g][h][j][i] = std::stof(num);
        }
      }
    }
  }

  for (int k = 0; k < model.size_of_map_; k++) {
    std::string line;
    std::getline(is, line);
    model.map_of_priors_[k] = std::stof(line);
  }

  return is;
}

const size_t Model::SizeOfImage(std::string path) {
  std::ifstream input_file(path);
  std::string label;
  size_t pointer = 1;

  while (getline(input_file, label)) {

    std::string send;

    while (getline(input_file, send)) {
      size_t size = send.size();
      size_ = size;

      if (pointer == size) {
        break;
      } else {
        pointer++;
      }
    }
    input_file.close();
  }
  return pointer;
}

size_t Model::CalculateClassLikelihood(Image image) {

  std::vector<std::vector<int>> predict_image = image.GetPixels();
  likelihood_vector_ = std::vector<float>(size_of_map_);

  for (size_t label = 0; label < size_of_map_; label++) {
    likelihood_vector_[label] += log(map_of_priors_[label]);
    for (size_t i = 0; i < size_; i++) {
      for (size_t j = 0; j < size_; j++) {
        likelihood_vector_[label] += log(probability_matrix_[label][predict_image[j][i]][j][i]);
      }
    }
  }

  size_t max = 0;

  for (size_t label = 1; label < size_of_map_; label++) {
    if (likelihood_vector_[max] < likelihood_vector_[label]) {
      max = label;
    }
  }

 return max;
}

float Model::CalculateAccuracy(std::vector<Image> images) {
  float sum = 0;
  size_t i;
  for (i = 0; i < images.size(); i++) {
    if(CalculateClassLikelihood(images[i]) == images[i].GetLabel()) {
      sum++;
    }
  }
  return sum/images.size() * 100;
}

const std::vector<Image>& Model::GetVectorOfImages() const { return images_; }

const std::map<size_t, float>& Model::GetMapOfPriors() const { return map_of_priors_; }

const std::vector<std::vector<std::vector<std::vector<float>>>>&
Model::GetProbabilityMatrix() const { return probability_matrix_; }

size_t Model::GetSizeOfMap() const { return size_of_map_; }

size_t Model::GetSize() const { return size_; }

float Model::GetKSmoothingConstant() const { return kSmoothingConstant; }


}  // namespace naivebayes
