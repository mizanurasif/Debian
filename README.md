
<p>
1.	What is the main objective of unsupervised learning?

A)	To learn a mapping from inputs to outputs
B)	To classify data into predefined categories
C)	To discover hidden patterns in data
D)	To maximize rewards in an environment


2.	What is true for Unsupervised Learning?

A)	It predicts target variable based on input features
B)	It discovers hidden pattern in unlabeled data
C)	It discovers patterns in labeled data
D)	It improves decision making through minimizing rewards


3.	Which of the following is type of Unsupervised learning?

 
A)	Clustering
B)	Regression
C)	Association
D)	Classification
 


4.	A marketing team wants to segment their customer base into distinct groups based on their purchasing behavior. Which type of machine learning would be most suitable for this task?

 
A)	Unsupervised Learning
B)	Supervised Learning
C)	Reinforcement Learning
D)	Regression
 

Ref: https://www.geeksforgeeks.org/unsupervised-learning/
Unsupervised learning is a branch of machine learning that deals with unlabeled data. Unlike supervised learning, where the data is labeled with a specific category or outcome, unsupervised learning algorithms are tasked with finding patterns and relationships within the data without any prior knowledge of the data’s meaning. Unsupervised machine learning algorithms find hidden patterns and data without any human intervention.

There are mainly 3 types of Algorithms which are used for Unsupervised dataset.
Clustering: Clustering in unsupervised machine learning is the process of grouping unlabeled data into clusters based on their similarities.
Association Rule Learning: Association rule learning is a common technique used to discover associations. This technique is a rule-based ML technique that finds out some very useful relations between parameters of a large data set.
Dimensionality Reduction: Dimensionality reduction is the process of reducing the number of features in a dataset while preserving as much information as possible.

Unsupervised learning has diverse applications across industries and domains. Key applications include:
Customer Segmentation: Algorithms cluster customers based on purchasing behavior or demographics, enabling targeted marketing strategies.
Anomaly Detection: Identifies unusual patterns in data, aiding fraud detection, cybersecurity, and equipment failure prevention.
Recommendation Systems: Suggests products, movies, or music by analyzing user behavior and preferences.



5.	Which approach would be appropriate for categorizing weather like Hot, Humid, Medium, Cold etc.?

 
A)	Clustering
B)	Regression
C)	Association
D)	Classification
 



6.	Which of the following best describes overfitting in machine learning?

A)	Model that performs poorly on both training and test data.
B)	Model that performs well on training data but poorly on test data.
C)	Model that is too simple to capture the underlying patterns in the data.
D)	Model that is just right, balancing complexity and generalization

7.	Which of the following conclusions can be derived from below data?
Training accuracy => 96%
Test accuracy =>71%
Validation accuracy => 70%
A)	The model is underfitting since training accuracy is high
B)	The model is overfitting since training accuracy is much higher than validation and test accuracy
C)	The model is generalizing well because all three accuracies are identical
D)	The validation and test accuracy should always be higher than training accuracy

 
8.	Suppose you have been given the following scenario for training and validation error for Linear Regression.
Scenario	Learning Rate	Number of iterations	Training Error	Validation Error
1	0.1	1000	100	110
2	0.2	600	90	105
3	0.3	400	110	110
4	0.4	300	120	130
5	0.4	250	130	150
     Which of the following scenario would give you the right hyperparameter?
 
A)	1
B)	2
C)	3
D)	4
 
9.	Which scenario describes Underfitting?

 
A)	High Bias and Low Variance
B)	High Variance and Low Bias
C)	Low Bias and Low Variance
D)	High Bias and High Variance
 

10.	Which of the below graph indicates Underfitting in Regression?

 
A)	Graph 1 
B)	Graph 2
C)	Graph 3
D)	Graph 1 & Graph 2
 


Ref: https://www.geeksforgeeks.org/underfitting-and-overfitting-in-machine-learning/
Bias: is the error that happens when a machine learning model is too simple and doesn’t learn enough details from the data. It’s like assuming all birds can only be small and fly, so the model fails to recognize big birds like ostriches or penguins that can’t fly and get biased with predictions.
•	High bias typically leads to underfitting, where the model performs poorly on both training and testing data because it fails to learn enough from the data.
•	Example: A linear regression model applied to a dataset with a non-linear relationship.

Variance: Error that happens when a machine learning model learns too much from the data, including random noise.
•	High variance typically leads to overfitting, where the model performs well on training data but poorly on testing data.

Overfitting happens when a model learns too much from the training data, including details that don’t matter (like noise or outliers).
Reasons for Overfitting:
1.	High variance and low bias.
2.	The model is too complex.
3.	The size of the training data.

Overfitting models are like students who memorize answers instead of understanding the topic. They do well in practice tests (training) but struggle in real exams (testing).

Reasons for Underfitting:
1.	The model is too simple, So it may be not capable to represent the complexities in the data.
2.	The input features which is used to train the model is not the adequate representations of underlying factors influencing the target variable.
3.	The size of the training dataset used is not enough.
4.	Excessive regularization are used to prevent the overfitting, which constraint the model to capture the data well.
5.	Features are not scaled.

Underfitting models are like students who don’t study enough. They don’t do well in practice tests or real exams. Note: The underfitting model has High bias and low variance.


An ideal model strikes a balance with low bias and low variance, capturing the overall pattern without overreacting to noise.
 


11.	A dataset contains numerical features with different scales. One feature ranges from 1 to 10, while another ranges from 1,000 to 10,000. Which preprocessing technique should be applied to ensure fair treatment of features in distance-based algorithms?

 
A)	One-hot encoding
B)	Feature scaling
C)	Dimensionality reduction
D)	Data augmentation
 

12.	What is not the purpose of feature scaling?

A)	Fair contribution of each feature
B)	Avoiding bias in distance calculation
C)	Optimize performance of calculation
D)	Reduce high correlation with particular feature

13.	What technique is used to scale numeric data to a fixed range [0,1] ?

 
A)	Standardization
B)	Normalization
C)	Encoding
D)	Splitting
 


14.	An e-commerce company wants to predict the likelihood of a customer making a repeat purchase. The dataset includes customer demographics, purchase history, and browsing behavior. The data scientist creates a new feature that represents the average time between purchases for each customer. What is this process called?

 
A)	Data Augmentation
B)	Standardization
C)	Feature Engineering
D)	Normalization
 


15.	Which of the following is an example of creating new features from existing ones?

 
A)	Dropping irrelevant columns
B)	Normalizing data
C)	Combining multiple features into one
D)	Shuffling the dataset
 

Ref: https://www.geeksforgeeks.org/what-is-feature-engineering/

Feature Engineering is the process of creating new features or transforming existing features to improve the performance of a machine-learning model.
Feature engineering in Machine learning consists of mainly 5 processes: Feature Creation, Feature Transformation, Feature Extraction, Feature Selection, and Feature Scaling.
Types of Feature Transformation:
1.	Normalization: Rescaling the features to have a similar range, such as between 0 and 1, to prevent some features from dominating others.
2.	Scaling: Scaling is a technique used to transform numerical variables to have a similar scale, so that they can be compared more easily. Rescaling the features to have a similar scale, such as having a standard deviation of 1, to make sure the model considers all features equally.
3.	Encoding: Transforming categorical features into a numerical representation. Examples are one-hot encoding and label encoding.
4.	Transformation: Transforming the features using mathematical operations to change the distribution or scale of the features. Examples are logarithmic, square root, and reciprocal transformations.

Types of Feature Extraction:
1.	Dimensionality Reduction: Reducing the number of features by transforming the data into a lower-dimensional space while retaining important information. Examples are PCA and t-SNE.
2.	Feature Combination: Combining two or more existing features to create a new one. For example, the interaction between two features.
3.	Feature Aggregation: Aggregating features to create a new one. For example, calculating the mean, sum, or count of a set of features.
4.	Feature Transformation: Transforming existing features into a new representation. For example, log transformation of a feature with a skewed distribution.

Why Feature Scaling?
1.	Improves Model Performance: By transforming the features to have a similar scale, the model can learn from all features equally and avoid being dominated by a few large features.
2.	Increases Model Robustness: By transforming the features to be robust to outliers, the model can become more robust to anomalies.
3.	Improves Computational Efficiency: Many machine learning algorithms, such as k-nearest neighbors, are sensitive to the scale of the features and perform better with scaled features.
4.	Improves Model Interpretability: By transforming the features to have a similar scale, it can be easier to understand the model’s predictions.


16.	Which value should you use in the missing box using median imputation? 
       Here is the marks of some students:
67		67	56	58	48	89		74

 
A)	67
B)	55
C)	51
D)	65
 

17.	You are working on a customer churn prediction project for a telecom company. You have collected a dataset containing customer information, such as age, gender, location. However, you notice that some of the data is missing. Specifically, 10% of the "age" column is missing, and 5% of the "location" column is missing. You decide to handle the missing data using imputation techniques. Which of the following imputation techniques would be most suitable for handling the missing data in this scenario?

A)	Mean imputation for both "age" and "location" columns 
B)	Median imputation for "age" column and KNN imputation for "location" column 
C)	Regression imputation for both "age" and "location" columns
D)	Multiple imputation for both "age" and "location" columns

Ref: https://blog.mitsde.com/data-imputation-techniques-handling-missing-data-in-machine-learning/
Best Practices for Choosing an Imputation Method
•	Numerical Data: For numerical data, mean, median, or KNN imputation are commonly used.
•	Categorical Data: For categorical variables, mode imputation or KNN imputation works well.

18.	Suppose the horizontal axis is an independent variable and the vertical axis is a dependent variable. Which of the following offsets do we use in linear regression’s least square line fit?

 
 
A)	Vertical offsets
B)	Perpendicular offset
C)	Both, depending on the situation
D)	None of above
 


19.	As per this diagram of Decision Tree algorithm, which split will be considered for root node?

 
A)	Split 1
B)	Split 2
C)	Split 3
D)	Split 4

20.	In below case of Decision Tree, Which one should be split in Root Node?
Feature 1	Feature 2	Class
A	X	0
A	Y	1
B	X	0
B	X	0
B	Y	1






 
A)	Feature 1
B)	Feature 2
C)	Class
D)	Either Feature 1 or Feature 2
 

 
21.	In below case of Decision Tree, What is Entropy of “A” Feature?
Feature	Class
A	0
A	1
B	0
B	0
B	1

 
A)	0.5
B)	0.25
C)	1
D)	1.5
 

22.	In below case of Decision Tree, What is Entropy of “B” Feature?
Feature	Class
A	0
A	1
B	0
B	0
B	1

 
A)	0.5
B)	0.92
C)	1
D)	0

Ref: https://medium.com/codex/decision-tree-for-classification-entropy-and-information-gain-cd9f99a26e0d

Entropy is a measure of disorder or impurity in the given dataset.
For a dataset that has C classes and the probability of randomly choosing data from class, i is Pi. Then entropy E(S) can be mathematically represented as
 
If we have a dataset of 10 observations belonging to two classes YES and NO. If 6 observations belong to the class, YES, and 4 observations belong to class NO, then entropy can be written as below.
   
 
If all the 10 observations belong to 1 class then entropy will be equal to zero. Which implies the node is a pure node.
 
If both classes YES and NO have an equal number of observations, then entropy will be equal to 1.
 
The Information Gain measures the expected reduction in entropy. Entropy measures impurity in the data and information gain measures reduction in impurity in the data. The feature which has minimum impurity will be considered as the root node.
Information gain of a parent node can be calculated as the entropy of the parent node subtracted entropy of the weighted average of the child node.



23.	In SVM, What is true for support vector?

A)	They Define direction of hyperplane
B)	They are data points closest to hyperplane
C)	They are data points furthest to hyperplane
D)	They represent outliers

 
24.	In SVM, what can happen when Regularization Parameter is set to very low value?

 
A)	Margin becomes wider
B)	Margin becomes narrower
C)	Misclassification increases
D)	Misclassification decreases 


Ref: https://www.geeksforgeeks.org/support-vector-machine-algorithm/
The key idea behind the SVM algorithm is to find the hyperplane that best separates two classes by maximizing the margin between them. This margin is the distance from the hyperplane to the nearest data points (support vectors) on each side.
 

The equation for the linear hyperplane can be written as:
 
Here:
•	w is the normal vector to the hyperplane (the direction perpendicular to it).
•	b is the offset or bias term, representing the distance of the hyperplane from the origin along the normal vector 
The distance between a data point x_i and the decision boundary can be calculated as:
 
where ||w|| represents the Euclidean norm of the weight vector w. Euclidean norm of the normal vector w
 
For a linearly separable dataset, the goal is to find the hyperplane that maximizes the margin between the two classes while ensuring that all data points are correctly classified. This leads to the following optimization problem:
 
yi is the class label (+1 or -1) for each training instance.

In the presence of outliers or non-separable data, the SVM allows some misclassification by introducing slack variables ζ
The optimization problem is modified as:
 
Here:
•	C is a regularization parameter that controls the trade-off between margin maximization and penalty for misclassifications.
•	ζ are slack variables that represent the degree of violation of the margin by each data point.
Increasing C will penalize outliers more and as a result misclassification will decrease but margin will be narrower
Decreasing C will penalize outliers less and as a result misclassification will increase but margin will be wider



25.	In SVM, which technique can be used to handle non-linearly separable data?

 
A)	Linear Kernel
B)	Hard Margin Classification
C)	Soft Margin Classification
D)	Kernel Trick

Ref: https://www.geeksforgeeks.org/kernel-trick-in-support-vector-classification/
The kernel trick is a method used in SVMs to enable them to classify non-linear data using a linear classifier. By applying a kernel function, SVMs can implicitly map input data into a higher-dimensional space where a linear separator (hyperplane) can be used to divide the classes. This mapping is computationally efficient because it avoids the direct calculation of the coordinates in this higher space.
Several kernel functions can be used, each suited to different types of data distributions:

Linear Kernel: No mapping is needed as the data is already assumed to be linearly separable.
Polynomial Kernel: Maps inputs into a polynomial feature space, enhancing the classifier's ability to capture interactions between features.
Radial Basis Function (RBF) Kernel: Also known as the Gaussian kernel, it is useful for capturing complex regions by considering the distance between points in the input space.
Sigmoid Kernel: Mimics the behavior of neural networks by using a sigmoid function as the kernel.

Ref: https://spotintelligence.com/2024/05/06/support-vector-machines-svm/
 

26.	What is hyperparameter in K Nearest Neighbor algorithm?
A)	Number of categories
B)	Number of data points in each category
C)	Number of nearest Neighbors
D)	Type of distance calculation from data point


27.	Given the following image, which of the following is the most likely classification result when using KNN with k=3?

 








 
A)	Class A
B)	Class B
C)	Can’t determine without more information
D)	New Class C


Ref: https://www.geeksforgeeks.org/k-nearest-neighbours

K-Nearest Neighbors is also called as a lazy learner algorithm because it does not learn from the training set immediately, instead it stores the dataset and at the time of classification it performs an action on the dataset.
As an example, consider the following table of data points containing two features:

 
The new point is classified as Category 2 because most of its closest neighbors are blue squares. KNN assigns the category based on the majority of nearby points. Here, K is considered as 5.

 
 
28.	Which of the following is NOT a key concept of K-means clustering?
 
A)	Centroid
B)	Distance metric
C)	Hyper parameter K
D)	Kernel Trick

 

Ref: https://www.geeksforgeeks.org/k-means-clustering-introduction/
We are given a data set of items with certain features and values for these features (like a vector). The task is to categorize those items into groups. To achieve this, we will use the K-means algorithm. ‘K’ in the name of the algorithm represents the number of groups/clusters we want to classify our items into. This is a hyperparameter.
 
The algorithm will categorize the items into k groups or clusters of similarity. To calculate that similarity, we will use the Euclidean distance as a measurement. The algorithm works as follows:  
•	First, we randomly initialize k points, called means or cluster centroids.
•	We categorize each item to its closest mean, and we update the mean’s coordinates, which are the averages of the items categorized in that cluster so far.
•	We repeat the process for a given number of iterations and at the end, we have our clusters.


29.	What should be the best choice of no. of clusters based on the this results?
 
A)	1
B)	2
C)	3
D)	4
 
Ref: https://www.geeksforgeeks.org/silhouette-algorithm-to-determine-the-optimal-value-of-k/
a(i) – The number of data points in the cluster assigned to the ith data point. It gives a measure of how well assigned the ith data point is to it’s cluster
b(i) – It is defined as the average dissimilarity to the closest cluster which is not it’s cluster
The silhouette coefficient s(i) is given by:-
 
We determine the average silhouette for each value of k and for the value of k which has the maximum value of s(i) is considered the optimal number of clusters for the unsupervised learning algorithm.



30.	In reinforcement learning, which of the following best describes the role of the reward function? 
A)	It determines the initial state of the environment
B)	It sets the goal for the agent to achieve
C)	It defines the transition probabilities between states
D)	It provides feedback to the agent based on its actions

31.	What is the limitation of Q-Learning?
A)	It only works for supervised learning
B)	It doesn’t work for deterministic environment
C)	It has very large state-action spaces
D)	It cannot handle stochastic environment

32.	Which learning method would be best appropriate for a self driving car?

 
A)	Supervised Learning
B)	Unsupervised Learning
C)	Reinforcement Learning
D)	Semi-supervised Learning

Ref: https://www.geeksforgeeks.org/the-role-of-reinforcement-learning-in-autonomous-systems/
Machine learning has a branch where RL (reinforce­ment learning) thrives. It mimics be­havioral psychology where an agent interacts with an environment to maximize rewards over time.
RL involves the agent taking actions, getting feedback (rewards/penalties), and adjusting behavior. The goal? Optimizing long-term performance in a sequential decision-making setup.

Example of Self Driving Car:
Agent: The self-driving car is the agent. The ecological agent interacts first-hand with the environment by means of decision-making and receiving outcomes.
Environment: The road and everything on it, including other cars, pedestrians, traffic signals, and weather conditions, form the environment.
State: The particular situation of the water with respect to the environment is our state. This could include information like the car's speed, position in the lane, distance to nearby objects, and traffic light status.
Action: The plays that cars behave are the actions. Examples include accelerating, braking, turning, changing lanes, and maintaining position.
Result: The consequence is a result of action with the environment whereas the car.This translates to the reward signal the agent receives. The amount of the prize for a safe and smooth trip is high, meanwhile, the crash or near-miss inevitably makes the reward to be low (or even penalty).

Ref: https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/

Q-Table:
 
Let’s say that a robot has to cross a maze and reach the end point. There are mines, and the robot can only move one tile at a time. If the robot steps onto a mine, the robot is dead. The robot has to reach the end point in the shortest time possible.
The scoring/reward system is as below:
•	The robot loses 1 point at each step. This is done so that the robot takes the shortest path and reaches the goal as fast as possible.
•	If the robot steps on a mine, the point loss is 100 and the game ends.
•	If the robot gets power ⚡️, it gains 1 point.
•	If the robot reaches the end goal, the robot gets 100 points.


Steps of Q-learning:
 
Update Q-table:

 
One disadvantage in Q-learning is that it might need to handle large Q-table or large state-action spaces for complex scenario.
33.	Which is the following statement is TRUE?
A)	Learning rate is too low in figure 3
B)	Learning rate is too high in figure 1
C)	Leaning rate is too high in figure 3
D)	Learning rate is high in figure 2

34.	Suppose you use gradient descent to train a ML model, but the cost function increases instead of decreasing. What could be the reason?

A)	The learning rate is too low
B)	The learning rate is too high
C)	The function is already minimized
D)	The gradient is always positive

Ref: https://www.geeksforgeeks.org/what-is-gradient-descent/

1. If Learning rate is too small: The algorithm will take tiny steps during iteration and converge very slowly. This can significantly increases training time and computational cost especially for large datasets. This process is termed as vanishing gradient problem.
 


2. If Learning rate is too big: The algorithm may take huge steps leading overshooting the minimum of cost function without settling. It fail to converge causing the algorithm to oscillate. This process is termed as exploding gradient problem.

 


35.	What is a perceptron in the context of neural networks?

A)	A type of loss function 
B)	A type of optimization algorithm 
C)	A type of activation function 
D)	A type of artificial neural network


36.	What happens to the perceptron when an input is classified incorrectly during the training process? 

A)	The weights will remain unchanged.
B)	The weights will be adjusted to reduce the error.
C)	The learning rate will be adjusted automatically.
D)	The perceptron will switch to a different learning algorithm.


37.	What is the purpose of using multiple hidden layers in an MLP?

A)	To reduce overfitting
B)	To capture non-linear relationships
C)	To simplify the model architecture
D)	To increase the number of parameters

Ref: https://www.geeksforgeeks.org/what-is-perceptron-the-simplest-artificial-neural-network/#what-is-perceptron

Perceptron is a type of neural network that performs binary classification that maps input features to an output decision, usually classifying data into one of two categories, such as 0 or 1.
What is the difference between Perceptron and Multi-layer Perceptron?

The Perceptron is a single-layer neural network used for binary classification, learning linearly separable patterns. 
In contrast, a Multi-layer Perceptron (MLP) has multiple layers, enabling it to learn complex, non-linear relationships. MLPs have input, hidden, and output layers, allowing them to handle more intricate tasks compared to the simpler Perceptron.


38.	Which activation function is commonly used in the output layer for multi-class classification tasks?    

 
A)	ReLU
B)	Sigmoid
C)	Tanh
D)	Softmax
 

39.	In which situation should you NOT use Softmax?

 
A)	Multi-class classification
B)	When probabilities must sum to 1
C)	Binary classification
D)	None

 
40.	Which activation function matches best with the below figure?
 

 
A)	ReLU
B)	Leaky ReLU
C)	Sigmoid
D)	Softmax
 


Ref: https://www.geeksforgeeks.org/activation-functions-neural-networks/

Sigmoid Activation Function is characterized by ‘S’ shape. It is mathematically defined as A=1 / (1+e−x) . This formula ensures a smooth and continuous output that is essential for gradient-based optimization methods.
•	It allows neural networks to handle and model complex patterns that linear equations cannot.
•	The output ranges between 0 and 1, hence useful for binary classification.

Softmax function is designed to handle multi-class classification problems. It transforms raw output scores from a neural network into probabilities. It transforms raw output scores from a neural network into probabilities. It works by squashing the output values of each class into the range of 0 to 1, while ensuring that the sum of all probabilities equals 1.


41.	What happens to the gradient of sigmoid function when x is very large?

 
A)	It remains constant
B)	It approaches 1
C)	It approaches 0
D)	It becomes negative 
 
Ref: https://www.geeksforgeeks.org/derivative-of-the-sigmoid-function/

Issue with Sigmoid Function in Backpropagation
One key issue with using the sigmoid function is the vanishing gradient problem. When updating weights and biases using gradient descent, if the gradients are too small, the updates to weights and biases become insignificant, slowing down or even stopping learning.

 
The shades red region highlights the areas where the derivative 
is very small (close to 0).


 
42.	What is the primary difference between a CNN and RNN? 
    
A)	CNN is used for sequential data and RNN is used for Image Processing 
B)	CNN is better for Image Processing and RNN is Better for sequential data 
C)	CNN doesn’t have Hidden Layers but RNN has Hidden Layers
D)	CNN is always slower than RNN

43.	An engineer is working on a speech recognition system. The input data consists of audio signals, and the goal is to transcribe speech into text. What type of model should he use?   
 
 
A)	Convolutional Neural Network (CNN)
B)	Feedforward Neural Network
C)	Long Short-Term Memory (LSTM)
D)	Decision Tree
 

Ref: https://www.geeksforgeeks.org/introduction-to-recurrent-neural-network/
Feedforward Neural Networks (FNNs) process data in one direction from input to output without retaining information from previous inputs. This makes them suitable for tasks with independent inputs like image classification. However FNNs struggle with sequential data since they lack memory.
Recurrent Neural Networks (RNNs) solve this by incorporating loops that allow information from previous steps to be fed back into the network. This feedback enables RNNs to remember prior inputs making them ideal for tasks where context is important.

 
RNNs are used in various applications where data is sequential or time-based:
Time-Series Prediction: RNNs excel in forecasting tasks, such as stock market predictions and weather forecasting.
Natural Language Processing (NLP): RNNs are fundamental in NLP tasks like language modeling, sentiment analysis, and machine translation.
Speech Recognition: RNNs capture temporal patterns in speech data, aiding in speech-to-text and other audio-related applications.
Image and Video Processing: When combined with convolutional layers, RNNs help analyze video sequences, facial expressions, and gesture recognition.

Limitations of Recurrent Neural Networks (RNNs):
Vanishing Gradient: During backpropagation, gradients diminish as they pass through each time step, leading to minimal weight updates. This limits the RNN’s ability to learn long-term dependencies, which is crucial for tasks like language translation.
Exploding Gradient: Sometimes, gradients grow uncontrollably, causing excessively large weight updates that destabilize training. Gradient clipping is a common technique to manage this issue.

Ref: https://www.geeksforgeeks.org/understanding-of-lstm-networks/
Long Short-Term Memory is an advanced version of recurrent neural network (RNN) architecture that was designed to model chronological sequences and their long-range dependencies more precisely than conventional RNNs.
It has been so designed that the vanishing gradient problem is almost completely removed, while the training model is left unaltered.


44.	A company wants to automatically categorize customer support emails into "Technical Issues," "Billing Queries," or "General Feedback." Which machine learning approach is most suitable for this task?

 
A)	Regression analysis
B)	Multi-class text classification
C)	Binary text classification
D)	Unsupervised clustering
 

45.	A search engine ranks documents for the query "best laptops 2025" by giving higher scores to rare terms like "2025" while downplaying common words like "best."  Which technique is likely powering this ranking?

 
A)	Bag of Words
B)	TF-IDF
C)	Word2Vec
D)	Transformer
  
46.	What TF stands for in TF-IDF?

 
A)	Text Frame
B)	Text Formulation
C)	Term Frequency
D)	Term Forwarding
 

47.	Suppose a word appears in every document in a dataset. What would be its IDF value?

 
A)	High
B)	Zero
C)	Negative
D)	Infinity
 

TF-IDF combines two components: Term Frequency (TF) and Inverse Document Frequency (IDF).
Term Frequency (TF): Measures how often a word appears in a document. A higher frequency suggests greater importance. If a term appears frequently in a document, it is likely relevant to the document’s content. Formula:
 
Inverse Document Frequency (IDF): Reduces the weight of common words across multiple documents while increasing the weight of rare words. If a term appears in fewer documents, it is more likely to be meaningful and specific.
 


 
48.	A sentiment analysis tool processes the sentence "The movie was great, but the ending sucked" and understands the mixed emotions by looking at both directions of the text. Which model is best suited for this task?

 
A)	GPT
B)	BERT
C)	Bag of Words
D)	Word2Vec

Feature	GPT (Generative Pre-trained Transformer)	BERT (Bidirectional Encoder Representations from Transformers)
Core Architecture	Autoregressive, generative	Bidirectional, context-based
Training Approach	Predicts the next word in a sequence	Uses masked language modeling to predict words from context
Direction of Context	Unidirectional (forward)	Bidirectional (both forward and backward)
Primary Usage	Text generation	Text analysis and understanding
Generative Capabilities	Yes, designed to generate coherent text	No, focuses on understanding text not generating
Pre-training	Trained on large text corpora	Trained on large text corpora with masked words
Fine-tuning	Necessary for specific tasks	Necessary, but effective with fewer training examples
Output	Generates new text sequences	Provides contextual embeddings for various NLP tasks

 
49.	What is the purpose of the "attention mechanism" in neural machine translation (NMT)?

A)	To reduce the vocabulary size of the source language
B)	To assign varying importance to different parts of the source sentence when generating the target sentence
C)	To ensure that all words in the source sentence are translated in the same order
D)	To eliminate the need for bilingual training data
Ref: https://www.geeksforgeeks.org/ml-attention-mechanism/
An attention mechanism is an Encoder-Decoder kind of neural network architecture that allows the model to focus on specific sections of the input while executing a task. It dynamically assigns weights to different elements in the input, indicating their relative importance or relevance. By incorporating attention, the model can selectively attend to and process the most relevant information, capturing dependencies and relationships within the data.
It enables the model to focus on certain areas of the input data, giving more weight to crucial features and disregarding unimportant ones. Each input attribute is given a weight based on how important it is to the output in order to accomplish this.

50.	A multinational corporation uses a neural machine translation system for real-time translation of emails between English and Korean. Users report that idiomatic expressions are often translated too literally, leading to confusion. What is the best way to improve the translation of idiomatic expressions?

A)	Replace the neural model with a rule-based system.
B)	Enrich the training data with a diverse set of idiomatic expressions and their proper translations.
C)	Increase the size of the vocabulary used by the model.
D)	Reduce the model’s training time to avoid overfitting.

 
                                                                                     
Samsung R&D Institute Bangladesh  



1.	What is true for Reinforcement Learning?
A)	It learns from labelled input-output pair
B)	It identifies clusters and structures in data
C)	It receives rewards or penalties from external environment
D)	It memorizes historic pattern without feedback

2.	A dataset has a categorical column “Color” with values “Red”, “Green” and “Blue”. How One hot Encoding will work here?
A)	Replace Color with values based on data (e. g., Red=1, Green=2, Blue=3)
B)	Replace Color with values based on summation of values related to “Red”, “Green”, “Blue” (1+2+3 = 6)
C)	Creates 3 new columns with binary values (0 to 1)
D)	Assign some random values

3.	What can be strategy to overcome Under fitting in Regression?
A)	Reduce model complexity like Applying Linear Regression instead of Polynomial Regression 
B)	Increase model complexity like Applying Polynomial Regression instead of Linear Regression
C)	Apply Regularization technique
D)	Increase number of features, Training data with more variation

4.	What is disadvantage of Decision Tree?
A)	Can’t handle numeric feature
B)	Can’t handle categorical data
C)	Difficult to interpret
D)	Tend to over fit as depth increases

5.	Why it is crucial to split data into training, validation and test sets in machine learning?
A)	To reduce the overall size of the dataset, making training faster.
B)	To ensure the model learns the training data perfectly, maximizing accuracy.
C)	To accurately estimate the model’s performance on unseen data and prevent overfitting.
D)	To simplify the feature engineering process by working with smaller subsets of data.

6.	A dataset contains 100 features, but training a model with all features results in slow performance and overfitting. Which preprocessing technique can reduce dimensionality while retaining most information?
A)	Principal Component Analysis (PCA)
B)	One-Hot Encoding
C)	R-score Normalization
D)	Z-score Normalization

7.	Which scenario describes Overfitting?
A)	High Bias and Low Variance 
B)	High Variance and Low Bias
C)	Low Bias and Low Variance 
D)	High Bias and High Variance

8.	Which of the following is NOT a key characteristic of a Random Forest algorithm?
A)	It is an ensemble learning method.
B)	It constructs multiple decision trees during training
C)	It uses the same subset of features for each tree.
D)	It aggregates the predictions of individual trees to make a final prediction

9.	What is disadvantage of KNN?
A)	 Training time is much higher
B)	 Inference time is much higher
C)	 It requires huge space to create model
D)	It can be used for only classification, not regression

10.	What is a Multi-Layer Perceptron (MLP) in the context of neural networks?
A)	A single-layer neural network
B)	A type of optimization algorithm
C)	A neural network with multiple hidden layers
D)	A neural network with multiple activation functions

11.	A search engine employs BERT to better understand user queries and match them with relevant documents. The system accurately captures user intent even in complex questions. What feature of BERT most contributes to this improved understanding?
A)	Its unidirectional language modeling.
B)	 Its use of a bag-of-words representation.
C)	 Its bidirectional transformer architecture that captures context from both left and right.
D)	 Its reliance on TF-IDF scores for word weighting.

12.	You want to train a machine learning model with limited datasets. You want to ensure your model generalizes well on unseen data/real world data. What techniques can help you assess this?
A)	Training on the entire dataset
B)	Cross-validation
C)	Using a complex model
D)	Regularization

13.	What is the purpose of handling missing values in data preprocessing?
A)	To increase model complexity
B)	To reduce the number of features
C)	To improve data quality and prevent biased analysis
D)	To make the dataset smaller

14.	A data scientist is working on a dataset containing customer purchase data, which includes features like age, income, and total purchase amount. The values of these features vary significantly, with income ranging from a few thousand to several million. To ensure that all features contribute equally to the machine learning model, which preprocessing step should the data scientist apply?
A)	Data Augmentation
B)	Normalization
C)	One-Hot Encoding
D)	Feature Extraction

15.	After training with SVM, some False Negatives found. What can you do to minimize false negatives?
A)	Use Linear Kernel
B)	Switch to K-Nearest Neighbour algorithm
C)	Use higher value of Regularization Parameter
D)	Use lower value of Regularization Parameter

16.	Which of the following is true about the K-Means clustering algorithm?
A)	It is a supervised learning algorithm
B)	The number of clusters (K) must be defined before running the algorithm
C)	It always finds the global optimum solution
D)	It does not require an initial centroid selection

17.	Which optimization algorithm is commonly used to train Multi-Layer Perceptrons?
A)	K-Means Clustering
B)	K-Nearest Neighbor
C)	Gradient Descent
D)	Principal Component Analysis (PCA)

18.	A chatbot developer trains a model to understand that "king" and "queen" are related, and "dog" is closer to "puppy" than to "car." The model uses word vectors trained on a large text corpus. Which technique is being applied?
A)	Bag of Words
B)	TF-IDF
C)	Word2Vec
D)	BERT

19.	An autonomous robot is navigating a maze using Q-learning. It tends to repeatedly choose the same paths and misses finding the optimal route. How can we encourage the robot to explore the maze more thoroughly and avoid sticking to familiar paths?
A)	Decrease the learning rate to make learning slower
B)	Increase the exploration rate (epsilon) to promote more random actions
C)	Apply a higher reward to the familiar paths
D)	Increase the discount factor to value future rewards more

20.	Given the following corpus:
Document 1: “The cat sat on the mat.”
Document 2: “The dog lay on the rug.”
Document 3: “The cat lay on the sofa.”
If we calculate TF-IDF, which word is likely to have the highest score?
A)	The
B)	On
C)	Sofa
D)	Cat

21.	What happens if we remove the activation function from all layers of a deep neural network?
A)	The network becomes a complex non-linear model
B)	The network behaves like a single-layer linear model
C)	The network learns faster
D)	The network can still model complex patterns.

22.	What is true about K-Mean Clustering?
E)	K-means is extremely sensitive to cluster center initializations
F)	Bad initialization can lead to Poor convergence speed
G)	Bad initialization can lead to bad overall clustering
A) 1 and 3
B) 1 and 2
C) 2 and 3
D) 1, 2 and 3

23.	You are developing a classifier to predict whether a customer will purchase a product based on Age, Income.
After training using SVM, you found overfitting. What should be best approach now to handle it?
A)	Use higher value of Regularization Parameter
B)	Use lower value of Regularization Parameter
C)	Use polynomial kernel instead of linear kernel
D)	Increase complexity of features used for classification

24.	In below case, new data point (6,10) and (10,12) will be classified to which category in KNN?
Feature 1	Feature 2	Category
3	10	1
5	15	1
7	18	2
9	12	2
11	14	2

A)	Both (6,10) and (10,12) under Category 1
B)	Both (6,10) and (10,12) under Category 2
C)	(6,10) under Category 1 and (10,12) under Category 2
D)	(6,10) under Category 2 and (10,12) under Category 1

25.	Which of the following is a common hyperparameter in Random Forest?
A)	Learning rate
B)	Regularization parameter
C)	Type of Kernel
D)	Number of trees (n_estimators)

26.	What is the difference between normalization and standardization?
A)	Normalization scales data to a specific range, while standardization scales data to have a mean of 0 and a standard deviation of 1
B)	Normalization removes missing values, while standardization adds new features
C)	Normalization introduces noise, while standardization smoothens data
D)	Normalization and standardization are the same

27.	In reinforcement learning, which of the following best describes the concept of exploration vs exploitation?
A)	Exploitation refers to the agent's ability to learn from its mistakes, while exploration refers to its ability to adapt to new situations
B)	Exploration refers to the agent's tendency to try out new actions, while exploitation refers to its tendency to stick with actions that have worked well in the past
C)	Exploration and exploitation are two different types of rewards that the agent can receive
D)	Exploration and exploitation are two different types of penalties that the agent can receive

28.	In below case of Decision Tree, What is Entropy of “B” Feature?
Feature	Class
A	0
A	1
B	1
B	1
B	1

A)	0.5
B)	0.8
C)	1
D)	0

29.	How the updated weight calculated in each iteration? [W = weight, N = learning rate, G = gradient]
A)	W = W + N x G
B)	W = W – N x G
C)	W = W x N
D)	W = W + G

30.	A retail company uses text classification to label product reviews as “positive,” “negative,” or “neutral.” However, the “neutral” category is consistently underperforming in model accuracy. What is the most likely cause of this issue?
A)	The training dataset has an imbalanced class distribution with very few neutral reviews.
B)	The model architecture is too complex for the task.
C)	There are too many neutral reviews, causing confusion in classification.
D)	The model is overfitting on the “neutral” category.

31.	Which of the following components is unique to LSTM networks and helps them retain information over long sequences?
A)	Convolutional Layer
B)	Memory Cell
C)	Recurrent Connection
D)	Fully Connected Layer

32.	What is the main purpose of using n-gram features in text classification?
A)	To count individual letters in words.
B)	To capture sequences of words for better context.
C)	To shorten the text before analysis.
D)	 To remove punctuation from the text.

33.	Output of which can be some rule like "If X, then Y”?
A)	Clustering
B)	Regression
C)	Association
D)	Classification

34.	A research team is developing a Neural Network to classify images of various dog breeds. The dataset includes thousands of labeled images of different breeds. To enhance the NN’s ability to recognize breed-specific features, which layer in the NN should the team focus on tuning?
A)	Fully Connected Layer
B)	Convolutional Layer
C)	Recurrent Layer
D)	Pooling Layer

35.	A company deploys a French-to-English translator for legal documents. The phrase "contrat à durée indéterminée" is translated as "contract of indefinite duration" instead of the more natural "permanent contract." What should the developers prioritize to improve this?
A)	Fine-tuning the model on legal domain-specific data
B)	Increasing the model’s vocabulary size
C)	Switching to a rule-based translation system
D)	Reducing the model’s reliance on neural networks

36.	Which regularization technique reduces the number of features used by setting some coefficients to zero?
A)	L1 Regularization (Lasso)
B)	L2 Regularization (Ridge)
C)	Batch Normalization
D)	Early Stopping

37.	Which of the above graph indicates best desired result in Regression?

 
A)	Graph 1
B)	Graph 2
C)	Graph 3
D)	Graph 1 & Graph 2

38.	Given the input vector X = [-2, 1, 0, 5, -4], what will be the output after applying ReLU?
A)	[0,1,0,1,0]
B)	[-1,1,0,1,-1]
C)	[0,1,0,5,0]
D)	[-2,0,0,5,-4]

39.	How Bag of Words represented?
A)	Sequence of words in original order
B)	Tree structure of word relationship
C)	Vector indicating meaning of word
D)	Vector indicating frequency of words

40.	Which type of RNN architecture is commonly used to improve the network's ability to learn long-term dependencies?
A)	Simple RNN
B)	Convolutional RNN
C)	Long Short-Term Memory (LSTM)
D)	Feedforward Neural Network

41.	A company wants to use linear regression to predict house prices based on location (categorical variable with 3 categories: “Downtown”, “Suburbs”, “Rural”) and square footage (numerical). Which approach is best for handling the location variable?
E)	Use it as a numerical variable (e.g., assign values 1, 2, 3)
F)	Convert it into dummy variables (one-hot encoding)
G)	Drop the location variable and only use square footage
H)	Apply principal component analysis (PCA) to reduce dimensions

42.	What is main purpose of feature scaling?
A)	To reduce number of features
B)	To reduce correlation among features
C)	To improve equal contribution of different features
D)	To handle missing values

43.	What is the expected relationship between training, validation, and test accuracy in a well-generalizing model?
A)	Training accuracy > Validation accuracy > Test accuracy
B)	Test accuracy > Training accuracy > Validation accuracy
C)	Training accuracy = Validation accuracy = Test accuracy
D)	Validation accuracy > Test accuracy > Training accuracy

44.	What happens to the perceptron when an input is classified incorrectly during the training process?
A)	The weights will remain unchanged.
B)	The weights will be adjusted to reduce the error.
C)	The learning rate will be adjusted automatically.
D)	The perceptron will switch to a different learning algorithm.

45.	A data analyst wants to reduce the dimensionality of a large dataset while preserving important information. Which type of machine learning would be most suitable for this task?
A)	Unsupervised Learning
B)	Reinforcement Learning
C)	Supervised Learning
D)	Semi-supervised Learning

46.	Consider a dataset of customers classifying their interest on Product:
Age	Income	Interest
20	Xxx	High
30	Xxx	High
40	Xxx	Low
50	Xxx	Low

You first used SVM linear kernel to train but observed that decision boundary is not clearly separating classes.
What you should do to improve Decision Boundary?
A)	Use Soft Margin SVM with linear Kernel
B)	Use non-linear Kernel like RBF
C)	Use higher value of Regularization Parameter
D)	Use lower value of Regularization Parameter

47.	Which activation function is known for mitigating the vanishing gradient problem in deep neural networks?
A)	Sigmoid
B)	Tanh
C)	Softmax
D)	ReLU

48.	Which one is advantage of Deep Learning over traditional Machine Learning?
A)	It requires less computational power
B)	It is more interpretable than traditional ML models
C)	It doesn’t require labelled data
D)	It can extract features from input data

49.	A dataset contains an imbalanced class distribution where 95% of the samples belong to Class A and only 5% to Class B. What preprocessing technique can be applied to balance the dataset before training a classification model?
A)	Standardization
B)	Oversampling the minority class
C)	Removing outliers
D)	Feature scaling

50.	Which activation function outputs values in the range [-1, 1]?
A)	ReLU
B)	Sigmoid
C)	Tanh
D)	Softmax

Date: 03-Sep-2025 (Set A)
Samsung R&D Institute Bangladesh 
AI/ML Basic Skill Exam


Name: _______________________________    GEN ID: ________________   Dept.: _____________

	What is the purpose of training dataset in machine learning?
	To evaluate the final performance of the trained model.
	To provide unseen data to assess how well the model generalizes.
	To expose the model to various examples and patterns, enabling it to learn.
	To fine-tune the model’s hyperparameters and prevent overfitting.

	Which of the following is a common application of unsupervised learning?
	Email spam detection
	Image recognition
	Market basket analysis
	Stock price prediction

	A model that has low accuracy on both training and validation sets is likely:
	Overfitting
	Underfitting
	Generalizing well
	Performing optimally

	What is true for Supervised Learning?
	It identifies patterns in unlabeled data
	It modifies its structure dynamically without external feedback
	It learns from labelled data to make prediction
	It learns by interacting with environment and receiving reward

	Which of the following is type of supervised learning?
	Clustering
	Regression
	Association
	Classification

	Which learning method would be best appropriate for weather prediction based on features like Humidity, Temperature etc.?
	Supervised Learning
	Unsupervised Learning
	Reinforcement Learning
	Semi-supervised Learning

	Which are disadvantages of Deep Learning compared to traditional Machine Learning?
	It requires more computational power
	It is less interpretable than traditional ML models
	Its inference or prediction time is always more than that of traditional ML models 
	It can’t extract features from input data

	A researcher wants to classify images of animals into different species. They have enough labelled data. Which type of machine learning would be most suitable for this task?
	Unsupervised Learning
	Reinforcement Learning
	Supervised Learning
	Semi-supervised Learning

	A robot is learning to navigate a maze. It receives a reward when it reaches the exit and a penalty when it hits a wall. Which learning type is most appropriate?
	Supervised Learning
	Unsupervised Learning
	Reinforcement Learning
	Clustering

	A dataset has missing values in a numeric column that are not randomly distributed and large in numbers. What would be best approach to handle those missing values?
	Drop entire column
	Drop all rows having the missing values
	Replace missing value with Mean, Median etc.
	Assign a non-zero constant number

	In a dataset with highly correlated features, which technique is best to reduce redundancy? 
	Standardization
	Normalization
	One-Hot Encoding
	Principal Component Analysis

	When Label Encoding is most suitable?
	Categorical feature has many unique values
	Categorical feature has high correlation with other feature
	Categorical feature is not much important for target
	Categorical feature has a natural order

	A machine-learning engineer is building a predictive model using a dataset that includes house sizes in square feet and prices in dollars. The features have different units and ranges. To bring all features to a comparable scale, which preprocessing technique should be used?
	Data Augmentation
	Standardization
	Feature Engineering
	Principal Component Analysis (PCA)

	A dataset containing information about customers' transactions, including the transaction amount, date, and location. To improve the model's ability to predict customer behavior, which new feature could be created from the existing features?
	Customer ID
	Transaction Time of Day
	Feature Engineering
	Customer Name

	A machine-learning engineer wants to apply a classification algorithm to categorical data containing country names. Which preprocessing technique should be used?
	Standardization
	One-hot encoding
	Principal Component Analysis (PCA)
	Feature scaling

	What is feature extraction in machine learning? 
	Adding new features to a dataset
	Transforming raw data into a set of features
	Removing irrelevant features from a dataset
	SVM will automatically adjust for different feature scales

	Which regression technique is being used when the relationship between independent vs dependent variable is not linear?
	Logistic Regression
	Polynomial Regression
	K-means Clustering
	Support Vector Machine

	Which of the above graph indicates Overfitting in Regression?

	Graph 1 
	Graph 2 
	Graph 3
	Graph 1 & Graph 2

	What can be strategy to overcome Overfitting in Regression?
	Reduce model complexity like Applying Linear Regression instead of Polynomial Regression
	Increase model complexity like Applying Polynomial Regression instead of Linear Regression
	Apply Regularization technique
	Increase number of features, Training data with more variation

	What is Multicollinearity in Regression?
	Dependent variable is highly correlated to one particular feature
	Dependent variable is highly correlated to multiple features
	Independent variables are highly correlated with each other
	Model is nonlinear

	In a decision tree, what is the purpose of the leaf node?
	To represent the class label or value to be predicted
	To store the conditions for splitting the data
	To indicate the importance of a feature
	To represent the depth of the tree

	A hospital wants to predict whether a patient has a disease based on their symptoms and test results. Doctors need an interpretable model to understand decision-making.
Which model is the best choice and why?
	Decision Tree because it is interpretable and easy to visualize
	SVM because it finds the best hyperplane for classification
	Random Forest because it combines multiple trees
	Neural Networks because they can detect complex patterns

	What technique does a Random Forest use to introduce randomness and diversity among the individual decision trees it build?
	Bagging(Bootstrap Aggregating)
	Boosting
	Principal Component Analysis (PCA)
	K-Nearest Neighbors (KNN)

	What is benefit of Decision Tree?
	Easier to visualize and Interpret
	Training time is much less
	Inference time is much less
	Less chance of overfitting

	In KNN, what type of distance between data points mostly used?
	Cosine similarity
	Difference between average values
	Hamming distance
	Euclidean distance

	What type of Algorithms is KNN?
	Supervised
	Deep Learning
	Unsupervised
	Reinforcement

	Which is not a SVM Kernel?
	Linear Kernel
	Convolutional Kernel
	RBF Kernel
	Sigmoid Kernel

	After training with SVM, you noticed that some points are very close to decision boundary. What can you do to make the margin wider?
	Use RBF Kernel
	Remove some support vectors
	Use higher value of Regularization Parameter
	Use lower value of Regularization Parameter

	Which hyper parameter controls trade-off between maximizing margin and minimizing misclassification?
	Gamma
	Kernel Type
	Degree in polynomial Kernel
	Regularization Parameter

	In K-Means clustering, what is the primary criterion for assigning a data point to a cluster?
	The data point is assigned to the cluster with the maximum distance from the centroid
	The data point is assigned randomly to a cluster
	The data point is assigned to the cluster whose centroid is closest to it
	The data point is assigned based on class labels

	A university applies K-Means clustering to group students based on study time and GPA. The algorithm finds 3 clusters with the following centroids:
Cluster	Study Hours (per week)	GPA
1	5	2.5
2	20	3.5
3	40	3.9

A new student studies 15 hours per week and has a GPA of 3.0
Which cluster will they likely belong to?
	Cluster 1
	Cluster 2
	Cluster 3
	Cannot be determined

	What happens if a dataset is not linearly separable when using a single-layer perceptron?
	It converges to an optimal solution
	It oscillates without converging 
	It learns a non-linear decision boundary 
	It adjusts weights using backpropagation

	An MLP (Multi-layer perceptron) has 2 input neurons, 2 hidden neurons, and 1 output neuron. If the input X = [1, 0], and the hidden layer weights are:
W1 = ■(0.8&0.4@0.2&0.6)  and biases b1 = [0.1, 0.2], compute the weighted sum z for the second hidden neuron.
	0.9
	0.4
	1.0
	0.5

	Which variant of gradient descent uses only a single training example per update?
	Batch Gradient Descent
	Mini-Batch Gradient Descent
	Stochastic Gradient Descent
	Adam optimizer

	What is the mathematical definition of the ReLU activation function?
	F(x) = x
	F(x) = max(0, x)
	F(x) = 1/1+e-x 
	F(x) = ex  

	If you replace Sigmoid function with ReLU in a deep neural network, which problem is most likely to be reduced?
	Vanishing gradient problem 
	Overfitting
	Exploding gradient problem
	Under fitting

	SRBD wants to predict next year’s revenue based on historical sales data using NN. Should the output layer have an activation function?
	Yes, use ReLU
	Yes, use Sigmoid
	Yes, use Softmax 
	No, the output should be continuous

	Which learning algorithm is commonly used to train a perceptron? 
	Backpropagation 
	K-Means 
	Gradient Descent 
	Linear Regression

	Which layer in a CNN is responsible for detecting features such as edges and textures?  
	Fully Connected Layer
	Pooling Layer
	Recurrent Layer
	Convolutional Layer

	What is the function of the pooling layer in a CNN?  
	To increase the number of parameters
	To introduce non-linearity
	To reduce the spatial dimensions of the input
	To improve the model's accuracy

	An engineer is using a Convolutional Neural Network (CNN) to develop an image recognition system for detecting objects in real-time video streams. The input images vary in size and resolution. To ensure that the CNN can process the images efficiently, what preprocessing step should the engineer apply?    
	One-Hot Encoding
	Feature Extraction
	Resizing
	Data Augmentation

	What is the primary purpose of a Recurrent Neural Network (RNN)?    
	To process static images 
	To handle sequential data 
	To perform clustering analysis 
	To reduce dimensionality

	A messaging app uses a bag-of-words (BoW) model to filter out spam messages. Despite flagging many spam messages correctly, the system fails to detect some spam that uses creative phrasing. What is the most likely limitation of the BoW approach in this scenario?
	It cannot capture word order and context.
	It requires extensive labeled data.
	It overemphasizes rare words.
	It uses excessive computational resources.

	What is the primary objective of the Word2Vec algorithm?
	To classify text into positive or negative categories
	To assign weights to words based on their document frequency
	To generate dense vector representations capturing word semantics
	To predict the next sentence in a sequence  

	A customer service bot generates responses like "I’m sorry to hear that, how can I assist you?" based on user complaints. It predicts the next word in a sequence during training. Which model architecture is most likely used?
	BERT
	GPT
	Word2Vec
	TF-IDF  

	In text classification, what does the term "feature extraction" refer to?
	Converting raw text into a numerical representation for machine learning
	Removing stop words from the text before processing
	Labeling the dataset with positive or negative tags
	Splitting text into training and testing sets

	What is the key advantage of neural machine translation (NMT) over traditional statistical machine translation (SMT)?
	NMT relies solely on dictionaries without training data
	NMT captures contextual relationships using neural networks
	NMT requires less computational power than SMT
	NMT avoids the use of parallel corpora

	A multinational corporation uses a neural machine translation system for real-time translation of emails between English and Korean. User’s report that idiomatic expressions are often translated too literally, leading to confusion. What is the best way to improve the translation of idiomatic expressions?
	Replace the neural model with a rule-based system.
	Enrich the training data with a diverse set of idiomatic expressions and their proper translations.
	Increase the size of the vocabulary used by the model.
	Reduce the model’s training time to avoid overfitting.  

	What is the main idea behind Bag of Words?
	Storing word order in sentence
	Encode Word meaning
	Storing collection of words
	Represent syntax of word

	What is the key characteristic of Q-Learning?
	It requires a model of environment
	It only works of deterministic environment
	Its model is much complex
	It’s a model-free algorithm
	</p>
