# Deep Learning Course – IIT Madras (BS Data Science and Applications)

This repository contains assignments and Python code for the Deep Learning course offered as part of the B.S. Data Science and Applications program at IIT Madras.

---

##  Table of Contents

- [Course Overview](#course-overview)
- [Contents](#contents)
- [Getting Started](#getting-started)
- [Assignments & Practice](#assignments--practice)
- [Example Outputs](#example-outputs)
- [Additional Resources](#additional-resources)
- [License](#license)
- [Authors & Acknowledgments](#authors--acknowledgments)

---

## Course Overview

This is a comprehensive collection of assignments, coding exercises, and practical implementations to reinforce foundational and advanced topics in deep learning, specifically designed for learners at IIT Madras.

---

## Contents

The repository covers a variety of critical deep learning topics, including—but not limited to:

- **Gradient Descent Variants** (`GDVsMGDVsNAG.py`)
- **Neural Networks & Backpropagation** (`NeuralNet_with_backpropogation.py`)
- **Activation Functions** (`activation_functions.py`)
- **Word Embedding Techniques**:
  - CBOW (`CBOW.py`)
  - Skip-Gram & Negative Sampling (`SkipGram.py`, `SkipGramNegative.py`)
  - GloVe (`GloVe.py`)
  - Hierarchical Softmax (`HierarchicalSoftmax.py`)
  - Contrastive Estimation (`contrastiveestimation.py`)
- **Optimization & Comparisons** (`optimizer_comparision.py`)
- **SVD, PCA, and Autoencoders** (`SVDVsPCAVsAutoEncoders.py`)
- **Classic Algorithms & Demos**:
  - XOR Perceptron (`XORPerceptron.py`)
  - XOR with an RNN (`XORNN.py`)
- **Conceptual Demonstrations**: Bias vs Variance (`BiasandVariance.py`)
- **Supplementary Assignments**: PyTorch-based practice (`PyTorchBonusAssignment.py`, `PyTorchBonusAssignment2.py`, `PyTorchBonusAssignment3.py`)
- **Project Files**:
  - Pretrained weight files (`parameters.npz`, `parameters1.npz`, `parameters_w11.npz`)
  - Sample dataset (`preprocessed_yelp_data.csv`)
  - Weekly assignments (`week4Q3.py`, `week5Q7.py`, `week8Practice.py`, `week9.py`, `week10.py`, `week11.py`, `week12.py`)

---

## Getting Started

To explore and run the examples:

1. **Clone the repository**  
   ```bash
   git clone https://github.com/jaiswalvik/deeplearning.git
   cd deeplearning
2. **Install dependencies**
   ```bash
   pip install numpy torch matplotlib pandas

(Add other required libraries as needed, based on specific scripts.)

3. **Run a script**
   ```bash
   python NeuralNet_with_backpropogation.py

## Assignments & Practice

  - **Weekly Assignments**: Follow the weekX.py files for structured assignment guidance correlating to course lectures.

  - **Bonus Practice**: The PyTorch bonus assignments are optional extensions for deeper hands-on practice.

## Example Outputs

Here are some typical results you can expect when running the scripts:

**Gradient Descent Variants** (GDVsMGDVsNAG.py)
Plots showing the convergence differences between Gradient Descent, Momentum, and Nesterov Accelerated Gradient.

**Neural Network with Backpropagation** (NeuralNet_with_backpropogation.py)
Training loss vs. epochs plot with decreasing error curve.

**Bias and Variance Demo** (BiasandVariance.py)
Visualizations comparing underfitting (high bias), good fit, and overfitting (high variance).

**Word Embeddings** (CBOW, SkipGram, GloVe)
Learned vector representations for words and cosine similarity comparisons.

**Autoencoders vs PCA vs SVD** (SVDVsPCAVsAutoEncoders.py)
Reconstruction errors and comparison plots.

## Additional Resources

This repository is linked to the official lecture playlist, which you can watch to complement the materials and gain deeper understanding:       
[Deep Learning Course – IIT Madras Video Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM)

## License

This project is for educational use only as part of IIT Madras’ B.S. Data Science and Applications program.
Please refer to the course’s academic integrity policies before using any code directly in assignments.

## Authors & Acknowledgments

**Vikas Jaiswal** – Repository maintainer

**IIT Madras Faculty** – For course design and lectures

**PyTorch, NumPy, Matplotlib, Pandas** – Core libraries used
