#include <string>
#include <map>

#include "core/image.h"

namespace naivebayes {

class Model {
 public:


  size_t CalculateClassLikelihood(Image image);

  float CalculateAccuracy(std::vector<Image> images);

  /**
   * To train the model for the
   * @param path
   */
  const void TrainModel(std::string & path);

  /**
   * Overrides the << operator function.
   * @param os
   * @param model
   * @return string of the file input.
   */
  friend std::ostream& operator<<(std::ostream& os, const Model& model);

  /**
   * Overrides the >> operator function.
   * @param is
   * @param model
   * @return reads from file and saves in an instance of model.
   */
  friend std::istream& operator>>(std::istream& is, Model& model);

  const std::vector<Image>& GetVectorOfImages() const;

  const std::map<size_t, float>& GetMapOfPriors() const;

  const std::vector<std::vector<std::vector<std::vector<float>>>>&
  GetProbabilityMatrix() const;

  size_t GetSizeOfMap() const;

  size_t GetSize() const;

  float GetKSmoothingConstant() const;

 private:
  std::vector<Image> images_;
  std::map<size_t, float> map_of_priors_;
  std::vector<std::vector<std::vector<std::vector<float>>>> probability_matrix_;
  size_t size_of_map_;
  size_t size_ = 0;
  float const kSmoothingConstant = 1;
  std::vector<float> likelihood_vector_;

  /**
   * Makes a 4 - D vector of i,j positions and number of images with black, grey
   * and white color.
   */
  void ProbabilityMatrices();

  /**
   * Modifies the 4 - D vector to have Laplace Probability and probability of
   * the classes.
   */
  void LaplaceProbability();

  /**
   * returns the size of
   * @return
   */
  const size_t SizeOfImage(std::string path);


};

}  // namespace naivebayes
