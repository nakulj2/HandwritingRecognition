#include <core/model.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <string>

TEST_CASE("Checking vector of Images and Image class") {

  naivebayes::Model model_1;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/test_file_text.txt";
  model_1.TrainModel(path);



  SECTION("Size of Vector of Images") {
    REQUIRE(model_1.GetVectorOfImages().size() == 5);
  }

  SECTION("Checking label of first picture in vector of pictures") {
    REQUIRE(model_1.GetVectorOfImages()[0].GetLabel() == 0);
  }

  SECTION("Checking label of last picture in vector of pictures") {
    REQUIRE(model_1.GetVectorOfImages()[4].GetLabel() == 1);
  }

  SECTION("Checking picture of first Image in Image vector") {
    std::string check = "  ## \n#+  #\n+#  +\n #  #\n   +#\n";
    REQUIRE(check.compare(model_1.GetVectorOfImages()[0].GetPicture()) == 0);
  }

  SECTION("Checking picture of last Image in Image vector") {
    std::string check = "   #+\n   + \n    #\n  +  \n   ##\n";
    REQUIRE(check.compare(model_1.GetVectorOfImages()[4].GetPicture()) == 0);
  }

  SECTION("Checking if pixels are working properly") {
    std::vector<std::vector<int>> checker;
    std::vector<int>temp;
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(1);
    temp.push_back(1);
    temp.push_back(0);
    checker.push_back(temp);
    temp.clear();
    temp.push_back(1);
    temp.push_back(2);
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(1);
    checker.push_back(temp);
    temp.clear();
    temp.push_back(2);
    temp.push_back(1);
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(2);
    checker.push_back(temp);
    temp.clear();
    temp.push_back(0);
    temp.push_back(1);
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(1);
    checker.push_back(temp);
    temp.clear();
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(0);
    temp.push_back(2);
    temp.push_back(1);
    checker.push_back(temp);
    REQUIRE(checker == model_1.GetVectorOfImages()[0].GetPixels());
  }
}

TEST_CASE("Checking image processing for different size") {

  naivebayes::Model model_1;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/different_test_file.txt";
  model_1.TrainModel(path);



  SECTION("Size of Vector of Images") {
    REQUIRE(model_1.GetVectorOfImages().size() == 3);
  }

  SECTION("Checking label of first picture in vector of pictures") {
    REQUIRE(model_1.GetVectorOfImages()[0].GetLabel() == 0);
  }

  SECTION("Checking picture of first Image in Image vector") {
    std::string check = "+++\n# #\n+++\n";
    REQUIRE(check.compare(model_1.GetVectorOfImages()[0].GetPicture()) == 0);
  }

  SECTION("Checking if pixels are working properly") {
    std::vector<std::vector<int>> checker;
    std::vector<int>temp;
    temp.push_back(2);
    temp.push_back(2);
    temp.push_back(2);
    checker.push_back(temp);
    temp.clear();
    temp.push_back(1);
    temp.push_back(0);
    temp.push_back(1);
    checker.push_back(temp);
    temp.clear();
    temp.push_back(2);
    temp.push_back(2);
    temp.push_back(2);
    checker.push_back(temp);

    REQUIRE(checker == model_1.GetVectorOfImages()[0].GetPixels());
  }
}

TEST_CASE("Check probability values of model instance") {
  naivebayes::Model model_1;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/test_file_text.txt";
  model_1.TrainModel(path);

  SECTION("Check Map probability of class 1") {
    REQUIRE(model_1.GetMapOfPriors().at(1) == Approx(0.571429));
  }

  SECTION("Check Map probability of class 0") {
    REQUIRE(model_1.GetMapOfPriors().at(0) == Approx(0.428571));
  }

  SECTION("Check Smoothing Constant ") {
    REQUIRE(model_1.GetKSmoothingConstant() == 1);
  }

}

TEST_CASE("Checking the operator function") {

  naivebayes::Model model_1;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/test_file_text.txt";
  model_1.TrainModel(path);

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

  SECTION("Checking length of picture of loaded data is same as length of original model") {
    REQUIRE(model_2.GetSize() == model_1.GetSize());
  }

  SECTION("Checking length of picture of loaded data") {
    REQUIRE(model_2.GetSize() == 5);
  }

  SECTION("Checking if size of both maps is same") {
    REQUIRE(model_2.GetSizeOfMap() == model_1.GetSizeOfMap());
  }

  SECTION("Checking size of the map from loaded data") {
    REQUIRE(model_2.GetSizeOfMap() == 2);
  }

  SECTION("Check Map probability of class 1") {
    REQUIRE(model_2.GetMapOfPriors().at(1) == Approx(0.571429));
  }

  SECTION("Check Map probability of class 0") {
    REQUIRE(model_2.GetMapOfPriors().at(0) == Approx(0.428571));
  }

  SECTION("Check probabilitty matrices") {
    REQUIRE(model_2.GetProbabilityMatrix() == model_1.GetProbabilityMatrix());
  }
}

TEST_CASE("Throwing Arguments") {

  SECTION("File Does Not Exist") {
    naivebayes::Model model_1;
    std::string path =
        "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/test_file_.txt";
    REQUIRE_THROWS_AS(model_1.TrainModel(path), std::invalid_argument);
  }
  SECTION("File is not in correct Format") {
    naivebayes::Model model_1;
    std::string path =
        "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/wrong_format_text_file_.txt";
    REQUIRE_THROWS_AS(model_1.TrainModel(path), std::invalid_argument);
  }

  SECTION("File Does Not Exist while operator >> overloading") {

    naivebayes::Model model_1;
    std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/test_file_text.txt";
    model_1.TrainModel(path);
    std::ofstream output_file("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj/data/saved_data.txt");
    if (output_file.is_open()) {
      REQUIRE_THROWS_AS(output_file << model_1, std::invalid_argument);
    }

  }

  SECTION("File Does Not Exist while operator << overloading") {

    naivebayes::Model model_1;
    std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/test_file_text.txt";
    model_1.TrainModel(path);
    std::ofstream output_file("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/saved_data.txt");
    if (output_file.is_open()) {
      output_file << model_1;
       output_file.close();
    }

    naivebayes::Model model_2;
    std::ifstream inputfile("/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj/data/saved_data.txt");
    if (inputfile.is_open()) {
      REQUIRE_THROWS_AS(inputfile >> model_2, std::invalid_argument);
    }

  }
}

TEST_CASE("CalculateClassLikelihood function") {

  naivebayes::Model model_training;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/validation_file.txt";
  model_training.TrainModel(path);

  naivebayes::Model model_1;
  std::string path_1 = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/training_data.txt";
  model_1.TrainModel(path_1);

  SECTION("Checking for correct prediction with 5000 pictures") {
    REQUIRE(model_1.CalculateClassLikelihood(model_1.GetVectorOfImages()[0]) == 5);
  }

  SECTION("Checking for incorrect prediction with 5000 pictures") {
    REQUIRE(!(model_1.CalculateClassLikelihood(model_1.GetVectorOfImages()[5]) == 9));
  }
}

TEST_CASE("Checking Accuracy") {

  naivebayes::Model model_training;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/validation_file.txt";
  model_training.TrainModel(path);

  naivebayes::Model model_1;
  std::string path_1 = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/training_data.txt";
  model_1.TrainModel(path_1);

  naivebayes::Model model_2;
  std::string path_2 = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/validation_file.txt";
  model_2.TrainModel(path_2);

  SECTION("Accuracy pf Validation file") {
    REQUIRE(model_2.CalculateAccuracy(model_2.GetVectorOfImages()) > 70);
  }

  SECTION("Accuracy pf Training Model file") {
    REQUIRE(model_1.CalculateAccuracy(model_1.GetVectorOfImages()) > 70);
  }

}

TEST_CASE("Checking if probability matrix is working for every element of 4 - D vector") {

  naivebayes::Model model;
  std::string path = "/Users/nakuljain/Downloads/cinder_0.9.2_mac/my-project/naive-bayes-nakulj2/data/different_test_file.txt";
  model.TrainModel(path);

  SECTION("Checking the size of the image vector") {
    REQUIRE(model.GetVectorOfImages().size() == 3);
  }
  SECTION("Checking the dimension of the image") {
    REQUIRE(model.GetSize() == 3);
  }
  SECTION("Checking the number of labels") {
    REQUIRE(model.GetMapOfPriors().size() == 2);
  }
  SECTION("Checking the Probability matrix for label 0 and unshaded with Laplace Smoothing") {

    REQUIRE(model.GetProbabilityMatrix()[0][0][0][0] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][0][1] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][0][2] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][1][0] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][1][1] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][0][1][2] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][2][0] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][2][1] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][0][2][2] == Approx(0.3333333));
  }
  SECTION("Checking the Probability matrix for label 0 and shaded with Laplace Smoothing") {

    REQUIRE(model.GetProbabilityMatrix()[0][1][0][0] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][1][0][1] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][1][0][2] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][1][1][0] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][1][1][1] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][1][1][2] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][1][2][0] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][1][2][1] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][1][2][2] == Approx(0.3333333));
  }
  SECTION("Checking the Probability matrix for label 0 and partially shaded with Laplace Smoothing") {

    REQUIRE(model.GetProbabilityMatrix()[0][2][0][0] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][2][0][1] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][2][0][2] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][2][1][0] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][2][1][1] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][2][1][2] == Approx(0.3333333));
    REQUIRE(model.GetProbabilityMatrix()[0][2][2][0] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][2][2][1] == Approx(0.6666667));
    REQUIRE(model.GetProbabilityMatrix()[0][2][2][2] == Approx(0.6666667));
  }
  SECTION("Checking the Probability matrix for label 1 and unshaded with Laplace Smoothing") {

    REQUIRE(model.GetProbabilityMatrix()[1][0][0][0] == Approx(0.5));
    REQUIRE(model.GetProbabilityMatrix()[1][0][0][1] == Approx(0.5));
    REQUIRE(model.GetProbabilityMatrix()[1][0][0][2] == Approx(0.75));
    REQUIRE(model.GetProbabilityMatrix()[1][0][1][0] == Approx(0.75));
    REQUIRE(model.GetProbabilityMatrix()[1][0][1][1] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][0][1][2] == Approx(0.75));
    REQUIRE(model.GetProbabilityMatrix()[1][0][2][0] == Approx(0.75));
    REQUIRE(model.GetProbabilityMatrix()[1][0][2][1] == Approx(0.5));
    REQUIRE(model.GetProbabilityMatrix()[1][0][2][2] == Approx(0.5));
  }
  SECTION("Checking the Probability matrix for label 1 and shaded with Laplace Smoothing") {

    REQUIRE(model.GetProbabilityMatrix()[1][1][0][0] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][0][1] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][0][2] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][1][0] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][1][1] == Approx(0.75));
    REQUIRE(model.GetProbabilityMatrix()[1][1][1][2] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][2][0] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][2][1] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][1][2][2] == Approx(0.25));
  }
  SECTION("Checking the Probability matrix for label 1 and partially shaded with Laplace Smoothing") {

    REQUIRE(model.GetProbabilityMatrix()[1][2][0][0] == Approx(0.5));
    REQUIRE(model.GetProbabilityMatrix()[1][2][0][1] == Approx(0.5));
    REQUIRE(model.GetProbabilityMatrix()[1][2][0][2] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][2][1][0] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][2][1][1] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][2][1][2] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][2][2][0] == Approx(0.25));
    REQUIRE(model.GetProbabilityMatrix()[1][2][2][1] == Approx(0.5));
    REQUIRE(model.GetProbabilityMatrix()[1][2][2][2] == Approx(0.5));
  }
}
