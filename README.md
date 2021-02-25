# HCIA-AI-Course
> This course is jointly launched by Huawei and Chongqing University of Posts and Telecommunications, and Dalian University of Technology，matching the HCIA-AI V3.0（Released on September 17, 2020). Through this course, you will systematically understand the AI development history, the Huawei Ascend AI system, the full-stack all-scenario AI strategy，and the algorithms related to traditional machine learning and deep learning; TensorFlow and MindSpore. HCIA-AI V1.0 will be offline on June 30, 2021.

*This lesson below is not mine, the source and copyright are for Huawei.*

**Content list**
- [AI OverView](https://github.com/gabboraron/HCIA-AI-Course#ai-overview)
  - [Symbolism](https://github.com/gabboraron/HCIA-AI-Course#symbolism)
  - [Connectionism](https://github.com/gabboraron/HCIA-AI-Course#connectionism)
  - [Behaviorism](https://github.com/gabboraron/HCIA-AI-Course#behaviorism)
  - [Types of AI:](https://github.com/gabboraron/HCIA-AI-Course#types-of-ai)
    - [Strong AI](https://github.com/gabboraron/HCIA-AI-Course#strong-ai)  
    - [Weak AI](https://github.com/gabboraron/HCIA-AI-Course#weak-ai)
  - [Huawei's Full-Stack, All-Scenario AI Portfolio](https://github.com/gabboraron/HCIA-AI-Course#huaweis-full-stack-all-scenario-ai-portfolio)
  - [Tools](https://github.com/gabboraron/HCIA-AI-Course#tools)
- [Machine Learning Overview](https://github.com/gabboraron/HCIA-AI-Course#machine-larning-oerview)


## AI OverView
file: https://github.com/gabboraron/HCIA-AI-Course/blob/main/058d50de2f9511ebb7f5fa163e57590a_01%20AI%20Overview.pdf

> **AI:** A new technical science that focuses on there search and development of theories, methods, techniques, and application systems for simulating and extending human intelligence
> 
> **Machine learning:** A core research field of AI. It focuses on the study of how computers cano btain new knowledge or skills by simulating or performing learning behavior of human beings, and reorganize existing knowledge architecture to improve its performance.
> 
> **Deep learning:** Deep learning aims to simulate the human brain to interpret data such as images, sounds, and texts.

### Symbolism
- The cognitive process of human beings is the process of inference and operation of various symbols.
- Human being is a physical symbol system, and so is a computer.
- The core of AI lies in knowledge representation, knowledge inference, and knowledge application. Knowledge and concepts can be represented with symbols. Cognition is the process of symbol processing while inference refers to the process of solving problems by using heuristic knowledge and search.

### Connectionism
- The basis of thinking is neurons rather than the process of symbol processing
- Human brains vary from computers. A computer working mode based on connectionism is proposed to replace the computer working mode based on symbolic operation.

### Behaviorism
- Intelligence depends on perception and action.
- Intelligence requires no knowledge, representation, or inference. AI can evolve like human intelligence. Intelligent behavior can only be demonstrated in the real world through the constant interaction with the surrounding environment.

![AI history](https://www.korem.com/wp-content/uploads/2020/03/history-of-artificial-intelligence-graphic-1024x577.jpg)

### Types of AI:
#### Strong AI
> Holds that it is possible to create intelligent machines that can really reason and solve problems. Such machines are considered to be conscious and self-aware, can independently think about problems and work out optimal solutions to problems, have their own system of values and world views, and have all the same instincts as living things,
#### Weak AI
> Holds that intelligent machines cannot really reason and solve problems. These machines only look intelligent, but do not have real intelligence.

### Huawei's Full-Stack, All-Scenario AI Portfolio
- **Application enablement:** provides end-to-end services (ModelArts), layered APIs, and pre-integrated solutions
- **MindSpore:** supports the unified training and inference framework that is independent of the device, edge, and cloud
    - provides automatic parallel capabilities. Can run a gorithms on dozens or even thousands of AI computing nodes with only a few lines of description.
- **CANN:** (Compute Architecture for Neural Networks) a chip operator library and highly automated operator development tool.
    - A chip operators library and highly automated operator development toolkitOptimal development efficiency, in-depth optimization of the common operator library, and abundant APIs 
- **Ascend:** provides a series of NPU IPs and chips based on a unified, scalable architecture.
- **Atlas:** enables an all-scenario AI infrastructure solution that is oriented to the device, edge, and cloud based on the Ascend series AI processors and various product forms.

### Algorithmic Bias
Algorithmic biases are mainly caused by data biases.

### Tools
- Tensorflow2.0: TensorFlow 2.0 has been officially released. It integrates Keras as its high-level API, greatly improving usability.

## Machine Learning Overview
file: https://github.com/gabboraron/HCIA-AI-Course/blob/main/7a5857dc2f9611ebb7f5fa163e57590a_02%20Machine%20Learning%20Overview.pdf

>  Machine learning is a core research field of AI, and it is also a necessary knowledge for deep learning. 

### Machine Learning Algorithms
https://forum.huawei.com/enterprise/en/five-levels-to-know-machine-learning-algorithms/thread/690267-895
> Machine learning (including deep learning) is a study of learning algorithms. A computer program is said to learn from experience 𝐸 with respect to some class of tasks 𝑇 and performance measure 𝑃 if its performance at tasks in 𝑇, as measured by 𝑃, improves with experience 𝐸.

![Differences Between Machine Learning Algorithms and Traditional Rule-Based Algorithms
](https://forum.huawei.com/enterprise/en/data/attachment/forum/202101/25/195307umpen4eayawnz0xf.png?Differences%20between%20machine%20learning%20algorithms%20and%20traditional%20rule-based%20algorithms.PNG)

**Machine learning can deal with:**
- **assification:** computer program needs to specify which of the k categories some input belongs to. To accomplish this task, learning algorithms usually output a function 𝑓:𝑅^𝑛 → (1,2,…,𝑘). For example, the image classification algorithm in computer vision is developed to handle classification tasks
- **Regression:** For this type of task, a computer program predicts the output for the given input. Learning algorithms typically output a function 𝑓:𝑅^𝑛 → 𝑅. An example of this task type is to predict the claim amount of an insured person (to set the insurance premium) or predict the security price
- **lustering:** A large amount of data from an unlabeled dataset is divided into multiple categories according to internal similarity of the data. Data in the same category is more similar than that in different categories. This feature can be used in scenarios such as image retrieval and user profile management

### Machine Learning Classification
#### Supervised learning
> Obtain an optimal model with required performance through training and learning based on the samples of known categories. Then, use the model to map all inputs to outputs and check the output for the purpose of classifying unknown data. 
> 
> ![supervised learning](https://forum.huawei.com/enterprise/en/data/attachment/forum/202102/22/165027pu2brbhruaolhxlu.png?4.PNG)
> 
> - *How much will I benefit from the stock next week?*
> - *What's the temperature on Tuesday?*
> - *Will there be a traffic jam on XX road during the morning rush hour tomorrow?*
> - *Which method is more attractive to customers: 5 yuan voucher or 25% off?*

#### Unsupervised learning: 
> For unlabeled samples, the learning algorithms directly model the input datasets. Clustering is a common form of unsupervised learning. We only need to put highly similar samples together, calculate the similarity between new samples and existing ones, and classify them by similarity. 
> 
> ![Unsupervised learning](https://forum.huawei.com/enterprise/en/data/attachment/forum/202102/22/165158r8ochahaoam39tia.png?6.PNG)
> 
> - *Which audiences like to watch movies of the same subject? *
> - *Which of these components are damaged in a similar way?*

####  Semi-supervised learning: 
> In one task, a machine learning model that automatically uses a large amount of unlabeled data to assist learning directly of a small amount of labeled data. 
> 
> ![Semi-supervised learning](https://forum.huawei.com/enterprise/en/data/attachment/forum/202102/22/165027pu2brbhruaolhxlu.png?4.PNG)
> - *the labaled data predict a high temperature, but this can be because other complications, and we can decide which is true*

####  Reinforcement learning: 
> Dinamic programing for interacting the enviroment.
> - always looks for best behaviors
> - Reinforcement learning is targeted at machines or robots. 
> 
> - *Autopilot: Should it brake or accelerate when the yellow light starts to flash? *
> - *Cleaning robot: Should it keep working or go back for charging?*

### Dataset
- Each data record is called a sample
- events that reflect the performance or  nature of a sample in particular aspects are called features

### Training set
- each sample is referred to as a training sample
- creating model from data = learning

### Test set
- the process of using the model obtained after learning for prediction
- each sample is called a test sample

### Data Processing
- more: https://forum.huawei.com/enterprise/en/important-concepts-of-machine-learning/thread/700585-893?from=latestPostsReplies
- Before using the data, you need to preprocess the data. There is no standard process for data preprocessing. Data preprocessing varies according to tasks and data set attributes. Common data preprocessing processes include removing unique attributes, processing missing values, encoding attributes, standardizing and regularizing data, selecting features, and analyzing principal components.

![Importance of Data Processing](https://forum.huawei.com/enterprise/en/data/attachment/forum/202102/23/140922brtbtgtmvghwzodb.png?13.PNG)

#### Data Cleansing
> Fill in missing values, and detect and eliminate causes of dataset exceptions.

In most cases, the collected data can be used by algorithms only after being preprocessed. The preprocessing operations include the following:
- Data filtering.
- Processing of lost data.
- Processing of possible exceptions, errors, or abnormal values.
- Combination of data from multiple data sources.
- Data consolidation.
- Generally, real data may have some quality problems.
- Incompleteness: contains missing values or the data that lacks attributes
- Noise: contains incorrect records or exceptions.
- Inconsistency: contains inconsistent records.


![Dirty Data](https://forum.huawei.com/enterprise/en/data/attachment/forum/202102/23/141005wquuvu8hbkzbb3qh.png?12.PNG) ![Workload of Data Cleansing](https://forum.huawei.com/enterprise/en/data/attachment/forum/202102/23/141059x8wavtutftaev708.png?14.PNG)

- With respect to classification, category data is encoded into a corresponding numerical representation.
- Value data is converted to category data to reduce the value of variables (for age segmentation).
- Other data
  - In the text, the word is converted into a word vector through word embedding (generally using the word2vec model, BERT model, etc).
  - Process image data (color space, grayscale, geometric change, Haar feature, and image enhancement)
- Feature engineering
  - Normalize features to ensure the same value ranges for input variables of the same model.
  - Feature expansion: Combine or convert existing variables to generate new features, such as the average.

#### Necessity of Feature Selection
> Generally, a dataset has many features, some of which may be redundant or irrelevant to the value to be predicted.
> 
> Feature selection is necessary in the following aspects:
> - Simplify models to make them easy for users to interpret
> - Reduce the training time
> - Avoid dimension explosion
> - Improve model generalization and avoid overfittingű

#### Feature Selection Methods - Filter
> Filter methods are independent of the model during feature selection.


