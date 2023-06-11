# Handwriting Recognition with Naive Bayes Machine Learning Model

Aim: To utilize a Naive Bayes classifier to recognize numbers from images. Handwriting recognition, which involves interpreting handwritten text as characters, will be the focus. Naive Bayes classifiers, based on Bayes' theorem, calculate the probabilities of an instance belonging to a specific class by considering the values of relevant features. In this scenario, we aim to determine whether an image belongs to the class of digits 0 to 9 by analyzing the state of the pixels in the images. Moreover, it contains the implementation of a Naive Bayes classifier connected to a sketchpad implemented in Cinder. The model classifies hand-drawn digits from 0 to 9 using the Naive Bayes algorithm and visualizes the classification results in real-time.

Accuracy:
The Naive Bayes classifier has been trained and tested on a dataset of 1000 images for validation/testing. The accuracy of the classifier on this dataset is approximately 78%.


File Structure:
src/ - Contains the source code files.
data/ - Contains the pre-labeled data files for validation/testing.
models/ - Contains the trained Naive Bayes models.
results/ - Contains the classification results and accuracy measurements.


Usage: 
Clone the repository and access the sketchpad with the following commands -> 
git clone https://github.com/nakulj2/HandwritingRecognition.git
cd naive-bayes-sketchpad
Install the required libraries

Run the application -> 
./run.sh

Use the sketchpad to draw a digit from 0 to 9.

Press the Enter key to classify the drawn digit.


