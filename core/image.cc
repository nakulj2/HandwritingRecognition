#include "core/image.h"

#include <iostream>

namespace naivebayes {

Image::Image(int label, std::string picture) {
  label_ = label;
  picture_ = picture;
  MakePixels();
}

Image::Image(const std::vector<std::vector<int>>& board) {
  pixels_ = board;
  label_ = 0;
  picture_ = "";
}

size_t Image::GetLabel() const { return label_; }

std::string Image::GetPicture() const { return picture_; }

std::vector<std::vector<int>> Image::GetPixels() const { return pixels_; }

Image::Image() = default;

void Image::MakePixels() {
  std::vector<int> line;

  for (size_t i = 0; i < picture_.size(); i++) {
    if (picture_[i] == '\n') {
      pixels_.push_back(line);
      line.clear();
    } else {
      if (picture_[i] == ' ') {
        line.push_back(0);
      } else if (picture_[i] == '#') {
        line.push_back(1);
      } else {
        line.push_back(2);
      }
    }
  }
}
}  // namespace naivebayes
