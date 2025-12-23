<b><p align="center">**Neural Networks & Machine Learning**</p> </b>


Here is my collection of my work focused on Machine Learning and Deep Learning. It covers a wide spectrum of tasks: from initial Exploratory Data Analysis (EDA), through building neural network components from scratch using NumPy, to implementing state-of-the-art architectures in PyTorch.

The project is divided into 7 main stages, each focusing on a different aspect of artificial intelligence:
<hr>
<b>Lab 1 Exploratory Data Analysis (EDA)</b>

Focus: Data understanding and preprocessing.

- Utilizes the UCI Heart Disease dataset.

- Performs statistical analysis (Shapiro-Wilk tests) and data visualization using Seaborn and Matplotlib.

- Covers class balancing, handling categorical variables (One-Hot Encoding), and correlation analysis.
<hr>
<b>Lab 2 Logistic Regression from Scratch</b>

Focus: Foundations of binary classification.

- Manual implementation of the Sigmoid function, Cross-Entropy Loss, and Gradient Descent.

- Comparison of training performance vs. model accuracy using custom weight updates.

<hr>
<b> Lab 3 Multilayer Perceptron (MLP) from Scratch </b>

Focus: Building a deep learning framework using only NumPy.

- Modular design with classes for Linear layers, ReLU, and Sigmoid activations.

- Implementation of Backpropagation and optimization steps.

- Conducts extensive "multi-testing" benchmarks across various learning rates, hidden layer sizes, and custom activation functions: f(x)=x/(1+|x|)​.
<hr>
<b>Lab 4 Neural Networks in PyTorch</b>

Focus: Transitioning to professional deep learning frameworks.

- Re-implementation of the MLP using torch.nn.

- Comparison of different optimizers: SGD (with momentum), Adam, and RMSprop.

- Analysis of learning curves (Train vs. Val Loss) across different batch sizes and learning rates.
<hr>
<b>Lab 5 Computer Vision with FashionMNIST</b>

Focus: Image classification and robustness.

- Classification of clothing items using the FashionMNIST dataset.

- Experiments on architectural depth (1-layer vs. 2-layer MLP).

- Noise Tolerance Analysis: Testing how Gaussian noise applied to training/testing images affects model accuracy and generalization.

<hr>
<b>Lab 6 Convolutional Neural Networks (CNN)</b>

Focus: Spatial feature extraction.

- Implementation of a FlexibleCNN for the MNIST digits dataset.

- Analysis of hyperparameters: Kernel Size (3×3 vs. 5×5), Number of Channels, and Pooling Size.

- Verification of CNN performance under noisy conditions.

<hr>
<b>Lab 7 Recurrent Neural Networks (RNN & LSTM)</b>

Focus: Natural Language Processing (NLP) and sequence modeling.

- Sentiment analysis on the IMDB Movie Reviews dataset.

- Implementation of a custom NLP pipeline: Tokenization, Vocabulary building, and Dynamic Padding.

- Comparison between Simple RNN and LSTM units.

- Study of the "Truncation Effect" – how limiting sequence length (20 vs. 50 words) impacts the understanding of sentiment.
