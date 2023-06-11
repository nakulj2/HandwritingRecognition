#include <core/image.h>
#include <core/model.h>
#include <visualizer/sketchpad.h>

namespace naivebayes {

namespace visualizer {

using glm::vec2;

Sketchpad::Sketchpad(const vec2& top_left_corner, size_t num_pixels_per_side,
                     double sketchpad_size, double brush_radius)
    : top_left_corner_(top_left_corner),
      num_pixels_per_side_(num_pixels_per_side),
      pixel_side_length_(sketchpad_size / num_pixels_per_side),
      brush_radius_(brush_radius) {

  board = std::vector<std::vector<int>>(num_pixels_per_side_,std::vector<int>(num_pixels_per_side_));

  naivebayes::Model model_1;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/training_data.txt";
  model_1.TrainModel(path);

  std::ofstream output_file("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/saved_data.txt");
  if (output_file.is_open()) {
    output_file << model_1;
    output_file.close();
  }
}

void Sketchpad::Draw() const {
  for (size_t row = 0; row < num_pixels_per_side_; ++row) {
    for (size_t col = 0; col < num_pixels_per_side_; ++col) {
      // Currently, this will draw a quarter circle centered at the top-left
      // corner with a radius of 20

      // TODO: Replace the if-statement below with an if-statement that checks
      // if the pixel at (row, col) is currently shaded
      if (board[row][col] == 1) {
        ci::gl::color(ci::Color("black"));
      } else if (board[row][col] == 2) {
        ci::gl::color(ci::Color::gray(0.3f));
      } else {
        ci::gl::color(ci::Color("white"));
      }

      vec2 pixel_top_left = top_left_corner_ + vec2(col * pixel_side_length_,
                                                   row * pixel_side_length_);

      vec2 pixel_bottom_right =
          pixel_top_left + vec2(pixel_side_length_, pixel_side_length_);
      ci::Rectf pixel_bounding_box(pixel_top_left, pixel_bottom_right);

      ci::gl::drawSolidRect(pixel_bounding_box);

      ci::gl::color(ci::Color("black"));
      ci::gl::drawStrokedRect(pixel_bounding_box);
    }
  }
}

void Sketchpad::HandleBrush(const vec2& brush_screen_coords) {
  vec2 brush_sketchpad_coords =
      (brush_screen_coords - top_left_corner_) / (float)pixel_side_length_;

  for (size_t row = 0; row < num_pixels_per_side_; ++row) {
    for (size_t col = 0; col < num_pixels_per_side_; ++col) {
      vec2 pixel_center = {col + 0.5, row + 0.5};

      if (glm::distance(brush_sketchpad_coords, pixel_center) <=
          brush_radius_) {
        // TODO: Add code to shade in the pixel at (row, col)
        if (board[row][col] == 0) {

        board[row][col] = 1;
        if (row != num_pixels_per_side_ - 1) {
          board[row + 1][col] = 2;
        }
        if (row != 0) {
          board[row - 1][col] = 2;
        }
        if (col != num_pixels_per_side_ - 1) {
          board[row][col + 1] = 2;
        }
        if (col != 0) {
          board[row][col - 1] = 2;
        }
      }
      }
    }
  }
}

void Sketchpad::Clear() {
  // TODO: implement this method
  board = std::vector<std::vector<int>>(num_pixels_per_side_,std::vector<int>(num_pixels_per_side_));
}

size_t Sketchpad::Classify() {

  naivebayes::Image image(board);

  naivebayes::Model model_2;
  std::ifstream inputfile("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/saved_data.txt");
  if (inputfile.is_open()) {
    inputfile >> model_2;
    inputfile.close();
  }

  return model_2.CalculateClassLikelihood(image);
}

}  // namespace visualizer

}  // namespace naivebayes
