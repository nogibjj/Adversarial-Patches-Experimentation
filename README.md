# Adversarial-Patches-Experimentation

## Project Motivation
As recent developments in machine learning and deep learning have paved the way for many technological advancements, these models have also become increasingly susceptible to potential adversarial attacks. Adversarial attacks are deliberate and malicious attempts to deceive these models by making subtle alterations to the input data to mislead the neural network into generating a different results. 

Adversarial patches are a type of adversarial attacks that involve placing a carefully crafted patch on an image to fool the model in making incorrect classifications. Drawing from Brown et al.’s work, we trained an adversarial patch on the CIFAR-10 dataset with the following goals:
- Measure the effectiveness of untargeted and targeted adversarial patches trained on ResNet-20. 
- Explore the sensitivity of the model to different patch sizes
- Assess the transferability of the targeted and untargeted adversarial patches on DenseNet and VGG-16.

## Methodology
Our overall approach is to train both targeted and untargeted adversarial patches of various sizes and test their transferability properties. The training of adversarial patches is as follows: 
Initialize a patch that that can be applied to any random locations on the image. 
Update the patch:

- Untargeted Approach: the patch is updated through stochastic gradient descent.
- Targeted Approach: labels of all images are assigned to the target class and the patch is updated through stochastic gradient ascent. The objective function takes the following form:

## Results

We found larger patches relative to input size exhibited higher success rates in deceiving neural networks, with notable transferability observed in untargeted attacks across DenseNet-121 and VGG-16. However, targeted attacks showed relatively limited transferability, highlighting challenges in carrying over adversarial patterns to different architectures.

## Conclusion
In this study, we utilized the CIFAR-10 dataset to train adversarial patches, aiming to examine both the impact of various patch types and the degree to which these trained patches could transfer across different Deep Neural Network architectures. Our experimentations with untargeted and targeted attacks were able to surpass baseline error rates highlighting their efficacy. Generally, we observed a positive correlation between patch size and attack success rate demonstrating that larger patches relative to the size of the input image have a better ability to deceive the neural network. Evaluations across DenseNet-121 and VGG-16 showcased notable transferability of adversarial patches for untargeted attacks. However, we observed relatively inferior transferability properties with targeted attacks using ‘Bird’ patch and ‘Horse’ patch as examples emphasizing nuanced challenges with targeted attacks in carrying over adversarial patterns to alternative architectures.

![image](https://github.com/nogibjj/Adversarial-Patches-Experimentation/assets/111402572/b3cfa71d-ff97-4a51-a024-6e0313f38be1)


