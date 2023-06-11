#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

#include <string>
#include <vector>

using std::string;

namespace naivebayes {

class Image {
 private:
  int label_;
  std::string picture_;
  std::vector<std::vector<int>> pixels_;

  /**
   * To make a 2 - D vector of pixels with 1 for black, 0 for white and 2 for
   * grey color.
   */
  void MakePixels();

 public:
  /**
   * Default constructor.
   */
  Image();

  Image(const std::vector<std::vector<int>> & board);

  /**
   * Constructor to initialize label and pixels for an image
   */
  Image(int label, std::string picture);

  /**
   * Returns label of a picture.
   * @return label
   */
  size_t GetLabel() const;

  /**
   * Returns a string of pixels.
   * @return vector of pixels
   */
  std::string GetPicture() const;

  /**
   * To get a 2 - D vector of pixels with 0,1, or 2 to represent vectors.
   * @return
   */
  std::vector<std::vector<int>> GetPixels() const;
};
}  // namespace naivebayes

#endif  // NAIVE_BAYES_IMAGE_H
