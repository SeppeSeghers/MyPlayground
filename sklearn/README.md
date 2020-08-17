# Introduction to Machine Learning

Nowadays Machine Learning is becoming more and more popular inside the development cycles. But what is ML, why we need and how we use it?

Let's start by giving a short definition. Machine learning is an application of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers learn automatically without human intervention or assistance and adjust actions accordingly.

In this topic we will focus on ML but while you are exploring you will probably come across with other applications such as Deep Learning, Artificial Intelligence, Data Science and Big Data. And now is the time to make sure you know which topic interfere in another.  

![](assets/AI-Venn-diagram.png)

Machine learning itself is often categorized into :
  - **Supervised Machine Learning algorithms** can apply what has been learned in the past to new data using labeled examples to predict future events. Starting from the analysis of a known training dataset, the learning algorithm produces an inferred function to make predictions about the output values. The system is able to provide targets for any new input after sufficient training. The learning algorithm can also compare its output with the correct, intended output and find errors in order to modify the model accordingly.
  
  
  - **Unsupervised Machine Learning algorithms** are used when the information used to train is neither classified nor labeled. Unsupervised learning studies how systems can infer a function to describe a hidden structure from unlabeled data. The system doesn’t figure out the right output, but it explores the data and can draw inferences from datasets to describe hidden structures from unlabeled data. 
  
  
  - **Semi-supervised Machine Learning algorithms** fall somewhere in between supervised and unsupervised learning, since they use both labeled and unlabeled data for training – typically a small amount of labeled data and a large amount of unlabeled data. The systems that use this method are able to considerably improve learning accuracy. Usually, semi-supervised learning is chosen when the acquired labeled data requires skilled and relevant resources in order to train it / learn from it. Otherwise, acquiring unlabeled data generally doesn’t require additional resources.
  
  
  - **Reinforsement Machine Learning algorithms** is a learning method that interacts with its environment by producing actions and discovers errors or rewards. Trial and error search and delayed reward are the most relevant characteristics of reinforcement learning. This method allows machines and software agents to automatically determine the ideal behavior within a specific context in order to maximize its performance. Simple reward feedback is required for the agent to learn which action is best; this is known as the reinforcement signal.

From each one of these four main types of machine learning there are unique techniques: 
    

![](assets/techniques.jpg)

## Why using ML?

Machine learning for enterprise use is accelerating, and not just at the periphery. Increasingly, businesses are putting machine learning applications at the center of their business models. The technology has enabled businesses to perform tasks at a scale previously unachievable, not only generating efficiencies for companies but also new [business opportunities](https://www.dezyre.com/article/top-10-industrial-applications-of-machine-learning/364). The growing use of machine learning in mission-critical business processes is reflected in the range of use cases where it plays an integral role. The following are examples:

- **Recommendation engines.** Most prominent online, consumer-facing companies today use recommendation engines to get the right product in front of their customer at the right time. Online retail giant Amazon pioneered this technology in the early part of the last decade, and it has since become standard technology for online shopping sites. These tools consider the browsing history of customers over time and match the preferences described by that history to other products the customer might not be aware of yet.


- **Fraud detection.** As more financial transactions move online, the opportunity for fraud has never been greater. That makes the need for fraud detection paramount. Credit card companies, banks and retailers are increasingly using machine learning applications to weed out likely cases of fraud. At a very basic level, these applications work by learning the characteristics of legitimate transactions and then scanning incoming transactions for characteristics that deviate. The tool then flags these transactions.


- **Customer analysis.** Most businesses today collect vast stores of data on their customers. This so-called big data includes everything from browsing history to social media activity. It's far too voluminous and diverse for humans to make sense of on their own. That's where machine learning comes in. Algorithms can troll the data lakes where enterprises store the raw data and develop insights about customers. Machine learning can even develop personalized marketing strategies that target individual customers and inform strategies for improving customer experience.


- **Financial trading.** Wall Street was one of the earliest adopters of machine learning technology, and the reason is clear: In a high-stakes world where billions of dollars are on the line, any edge is valuable. Machine learning algorithms are able to examine historical data sets, find patterns in stock performance and make predictions about how certain stocks are likely to perform in the future.


- **Virtual assistants.** By now, most people are familiar with virtual assistants from tech companies like Apple and Google. What they might not know is the extent to which machine learning powers these bots. Machine learning enters in a number of different ways, including deep learning, a machine learning technique based on neural networks. Deep learning plays an important role in developing natural language processing, which is how the bot is able to interact with the user, and in learning the user's preferences.


- **Self-driving cars.** This is where machine learning enters the realm of AI that aims to be on par with human intelligence. Autonomous vehicles use neural networks to learn to interpret objects detected by their cameras and other sensors, and to determine what action to take to move a vehicle down the road. In this way, machine learning algorithms can use data to come close to replicating human-like perception and decision-making.

These are just some examples, but there are countless more. Any business process that either produces or uses large amounts of data -- particularly structured, [labeled data](https://www.techopedia.com/definition/33695/labeled-data) -- is ripe for automation that uses machine learning. Enterprises across all industries have learned this and are working to implement machine learning methods throughout their processes.

It's not hard to see why machine learning has entered so many situations. Enterprises that have adopted machine learning are solving business problems and reaping value from this AI technique. Here are six main benefits:

    1.increased productivity

    2.lower labor costs

    3.better financial forecasting

    4.clearer understanding of customers

    5.fewer repetitive tasks for workers

    6.more advanced and human-like output.

## Challenges of using ML

The question is no longer whether to use machine learning, it's how to operationalize machine learning in ways that return optimal results. That's where things get tricky.

Machine learning is a complicated technology that requires substantial expertise. Unlike some other technology domains, where software is mostly plug and play, machine learning forces the user to think about why they are using it, who is building the tools, what their assumptions are and how the technology is being applied. There are few other technologies that have so many potential points of failure.

The **wrong use case** is the downfall of many machine learning applications. Sometimes enterprises lead with the technology, looking for ways to implement machine learning, rather than allowing the problem to dictate the solution. When machine learning is shoehorned into a use case, it often fails to deliver results.

The **wrong data** dooms machine learning models faster than anything. Data is the lifeblood of machine learning. Models only know what they've been shown, so when the data they train on is inaccurate, unorganized or biased in some way, the model's output will be faulty.

**Bias** frequently hampers machine learning implementations. The many types of bias that can undermine machine implementations generally fall into the two categories. One type happens when data collected to train the algorithm simply doesn't reflect the real world. The data set is inaccurate, incomplete or not diverse enough. Another type of bias stems from the methods used to sample, aggregate, filter and enhance that data. In both cases, the errors can stem from the biases of the data scientists overseeing the training and result in models that are inaccurate and, worse, unfairly affect specific populations of people. In his article "6 ways to reduce different types of bias in machine learning," analyst Ron Schmelzer explained the types of biases that can derail machine learning projects and how to mitigate them.

**Black box functionality** is one reason why bias is so prevalent in machine learning. Many types of machine learning algorithms -- particularly unsupervised algorithms -- operate in ways that are opaque, or as a "black box," to the developer. A data scientist feeds the algorithm data, the algorithm makes observations of correlations and then produces some sort of output based on these observations. But most models can't explain to the data scientist why they produce the outputs they do. This makes it extremely difficult to detect instances of bias or other failures of the model.

**Technical complexity** is one of the biggest challenges to enterprise use of machine learning. The basic concept of feeding training data to an algorithm and letting it learn the characteristics of the data set may sound simple enough. But there is a lot of technical complexity under the hood. Algorithms are built around advanced mathematical concepts, and the code that algorithms run on can be difficult to learn. Not all businesses have the technical expertise in house needed to develop effective machine learning applications.

**Lack of generalizability** prevents machine learning from scaling to new use cases in most enterprises. Machine learning applications only know what they've been explicitly trained on. This means a model can't take something it learned about one area and apply it to another, the way a human would be able to. Algorithms need to be trained from scratch for every new use case.

## Roadmap

Implementing machine learning is a multistep process requiring input from many types of experts. Here is an outline of the process in six steps.

- Any machine learning implementation starts with the **identification of a problem**. The most effective machine learning projects tackle specific, clearly defined business challenges or opportunities.
    
    
- Following the problem formulation stage, data science teams should **choose their algorithm**. Different machine learning algorithms are better suited for different tasks, as explained in this article on "9 types of machine learning algorithms" by TechTarget editor Kassidy Kelley. Simple linear regression algorithms work well in any use case where the user seeks to predict one unknown variable based on another known variable. Cutting-edge deep learning algorithms are better at complicated things like image recognition or text generation. There are dozens of other types of algorithms that cover the space between these examples. Choosing the right one is essential to the success of machine learning projects.
    
    
- Once the data science team identifies the problem and picks an algorithm, the next step is to **gather data**. The importance of collecting the right kind of and enough data is often underestimated, but it shouldn't be. Data is the lifeblood of machine learning. It supplies algorithms with everything they know, which in turn defines what they are capable of. Data collection involves complicated tasks like identifying data stores, writing scripts to connect databases to machine learning applications, verifying data, cleaning and labeling data and organizing it in files for the algorithm to work on. While these are tedious and complicated jobs, their importance cannot be overstated.
    
    
- Now it's time for the magic to begin. Once the data science team has all the data it needs, it can start **building the model**. This step in the machine learning process will differ substantially depending on whether the team is using a supervised machine learning algorithm or an unsupervised algorithm. When the training is supervised, the team feeds the algorithm data and tells it what features to examine. In an unsupervised learning approach, the team essentially turns the algorithm loose on the data and comes back once the algorithm has produced a model of what the data looks like. Learn how to build a neural network model in this expert tip.
    
    
- **Application development** is next. Now that the algorithm has developed a model of what the data looks like, data scientists and developers can build that learning into an application that addresses the business challenge or opportunity identified in the first step of the process. Sometimes this is very simple, like a data dashboard that updates sales projections based on changing economic conditions. It could be a recommendation engine that has learned to tailor its suggestions based on past customer behavior. Or it could be a component of cutting-edge medical software that uses image recognition technology to detect cancer cells in medical images. During the development stage, engineers will test the model against new, incoming data to make sure it delivers accurate predictions.
    
    
- Even though the primary work is complete, now is not the time to walk away from the model. The last step in the machine learning process is **model validation**. Data scientists should verify that their application is delivering accurate predictions on an ongoing basis. If it is, there's likely little reason to make changes. However, model performance typically degrades over time. This is because the underlying facts that the model trained on -- whether economic conditions or customer tendencies -- shift as time goes by. When this happens, the performance of models gets worse. This is the time when data scientists need to retrain their models. Here, the whole process essentially starts over again.

![](assets/ml_roadmap.jpg)

Before we go any further into our exploration , let's take a minute to define our terms. It is important to have an understanding of the vocabulary that will be used when describing Scikit-Learn's functions.

To begin with, a machine learning system or network takes inputs and outputs. The inputs into the machine learning framework are often referred to as **features** .

Features are essentially the same as variables in a scientific experiment, they are characteristics of the phenomenon under observation that can be quantified or measured in some fashion.

When these features are fed into a machine learning framework the network tries to discern relevant patterns between the features. These patterns are then used to generate the outputs of the framework/network.

The outputs of the framework are often called **labels**, as the output features have some label given to them by the network, some assumption about what category the output falls into.

## The work never ends

The management and maintenance of machine learning applications is one area that's sometimes given short shrift, but it can be what makes or breaks use cases.

The basic functionality of machine learning depends on models learning trends -- such as customer behavior, stock performance and inventory demand -- and projecting them to the future to inform decisions. However, underlying trends are constantly shifting, sometimes slightly, sometimes substantially. This is called concept drift, and if data scientists don't account for it in their models, the model's projections will eventually be off base.

The way to correct for this is to never view models in production as finished. They demand a constant state of verification, retraining and reworking to ensure they continue to deliver results.

**Verification.** Data scientists often will hold out a segment of new, incoming data and then verify the model's predictions to make sure they are close to the new, incoming data.


**Retraining.**If a model's results start to deviate significantly from actual observed data, it's time to retrain the model. Data scientists will need to source a completely new set of data that reflects current conditions.


**Rebuilding.** Sometimes the concept a machine learning model is supposed to predict will change so much that the underlying assumptions that went into the model are no longer valid. In these cases it may be time to completely rebuild the model from scratch.

## Platforms

The machine learning space features strong competition between open source tools and software built and supported by traditional vendors. Usually machine learning software from a vendor or open source tool is chosen as it is common for applications to be hosted in the cloud computing environments and delivered as a service. There are more vendors and platforms than one article could name, but the following list gives a high-level overview of offerings from some of the bigger players in the field.

### Vendor Tools

- Amazon Sagemaker is a cloud-based tool that allows users to work at a range of levels of abstraction. Users can run pretrained algorithms for simple workloads or code their own for more expansive applications.


- Google Cloud is a collection of services that range from plug-and-play AI components to data science development tools.IBM Watson Machine Learning is delivered through the IBM cloud and allows data scientists to build, train and deploy machine learning applications.


- Microsoft Azure Machine Learning Studio is a graphical user interface tool that supports building and deploying machine learning models on the Microsoft cloud.


- SAS Enterprise Miner is a machine learning offering from a more traditional analytics company. It focuses on building enterprise machine learning applications and productionalizing them quickly.


### Open source

- Caffe is a framework is specifically engineered to support the development of deep learning models -- in particular, neural networks.


- Scikit-learn is an open source library of Python code modules that allow users to do traditional machine learning workloads like regression analysis and clustering.


- TensorFlow is a machine learning platform built and open sourced by Google. It is commonly used for developing neural networks.


- Theano was originally released in 2007 and is one of the oldest and most trusted machine learning libraries. It is optimized to run jobs on GPUs, which can result in fast machine learning algorithm training.


- Torch is a machine learning library that is optimized to train algorithms on GPUs. It is built primarily to train deep learning neural networks.

And now I can see the terror in your eyes!"We will deal with all these? How do we know which is what? We will be confused for sure. Please take me out of here, I promise I will be a good person from now on". As you already know, learn by doing is the best way, so let's go for it! 

![Let's go!](https://media1.tenor.com/images/1d186fbcb3995bb71cceefa852862530/tenor.gif?itemid=3461551)
