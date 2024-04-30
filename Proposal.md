# cs171project

Title: Sea Animal Classification
Authors: Daniel Chang and Jesse Ge

Introduction: The topic of our project is classifying sea animals using images. This task is going to be finding patterns within images to figure out what animal is in the pictures. This is relevant because this can help with exploration of the ocean, sending robots and cameras to depths humans cannot go to, and having the ability to categorize and recognize each animal by itself. The dataset is called Sea Animals Image Dataset.
https://www.kaggle.com/datasets/vencerlanz09/sea-animals-image-dataste

Problem Statement: The problem we will solve is the identification of sea animals from images. In the real world, when scientists want to explore the ocean, they can send unmanned submarines to take pictures of sea animals, and be able to identify them by themselves. 

Data Description:  The dataset contains images of various different sea animals. These include Seahorse, Nudibranchs, Sea Urchins, Octopus, Puffers, Rays, Whales, Eels, Crabs, Squid, Corals, Dolphins, Seal, Penguin, Starfish, Lobster, Jelly Fish, Sea Otter, Fish, Shrimp, and Clams. The sea animal will be clearly seen in each image. The image sizes are a maximum of 300px width and height. It is possible that there are some images that are bad quality that we may have to take out of the dataset. 

Methodology: The machine learning technique we will use is convolutional neural networks. The pre-trained model we will use is ResNet50. We will use local connectivity without weight sharing, with single input channel, and single output map. Some libraries and frameworks we plan to use are tensorflow, keras, OpenCV, and Pandas.

Innovation and Objectives: An innovative aspect of our project would be the ability to identify sea creatures from photos. Our approach would be unique, because we will be specifically classifying sea creatures. We hope to create a model that can accurately identify what sea animal is in a picture. 

Evaluation Metrics: Our evaluation metrics contain the accuracy, precision, recall, and F1-score. These will remain relevant regardless of the model we use because it provides a basis for testing the success of the model. We will also utilize the confusion matrix to understand misclassifications.

Related Work: There are several studies that also include the classification of animals using a machine learning model. Many of these studies use datasets like CIFAR-10 and ImageNet, and have been deemed rather successful. Although accurate for the most part, some species look similar to others; this has caused error within models. Using our sea animal dataset, we hope to address this limitation.


Timeline:
Sprint 1: Data preprocessing/analyzation
Sprint 2: Model development and training
Sprint 3: Evaluation analyzation and documentation

Conclusion: The purpose of this project is to develop an accurate and efficient model that recognizes patterns within an image and classifies them within each species. We are specifically working with a sea animal image dataset; the success of this project may contribute to real world sea exploration, and can prove to be a valuable tool within ocean research.

References:

E. Newman, M. Kilmer, L. Horesh, Image classification using local tensor singular value decompositions (IEEE, international workshop on computational advances in multi-sensor adaptive processing. IEEE, Willemstad, 2018), pp. 1–5.

Chollet, Francois. "Building powerful image classification models using very little data." Keras Blog 5 (2016): 90-95.

Shalika, A. and Seneviratne, L. (2016) Animal Classification System Based on Image Processing & Support Vector Machine. Journal of Computer and Communications, 4, 12-21. doi: 10.4236/jcc.2016.41002.
