# Super NEAT Bros. Final Report

Aidan Albers, Jeremy Webb, Zachary Baskin, Zachary Minot

## Summary

![infographic](img/super-neat-bros-infographic.png)

## Introduction/Background

Games often have AI that play as Non-Player Characters (NPCs). They could be enemies or allies as simple
as the aliens in Space Invaders or as complex as the demons and monsters that fight each other in DOOM. However,
we thought,

"What if instead the AI *played* the game?"

To choose the game, we recognized that we should probably attempt an older game. Older games are much easier to both control
(due to often having less complicated and controls overall) and extract into data for our model
(due to often using tilesets).
Super Mario Bros. is a classic game for the Nintendo Entertainment System that has very few controls:
right, left, jump, and run. It also has world split into a tileset that allows us to easily view the game as a grid. Additionally,
the goal of Super Mario Bros. is as simple as getting to the flag at the right of the world. The game is run on an emulator on a PC for
both functionality and feasability. We've seen this done on Super Mario Bros by Sethbling, a Youtube content creator, but the specific implementation
he used was not very successful. We plan to adapt his algorithm and improve it to search for better results.

Although this is intriguing, we thought we could explore more in similar ideas. Speedrunners, people who
play games in attempt to complete them the fastest, not only memorize inputs but entire layouts of levels.
Often times they can recognize levels just based on a few key block placements or a certain enemy position.
This then lead us to ask the question,

"If the AI can play the game, can it then *memorize* it?"

Thus, the second problem of our project was born. Can a machine look at a the screen of Super Mario Bros. and
detect which level that screen is on? One trick is that there is a level counter on the game, so we would need to remove it.
At its whole, the artificial intelligence would be able to look at a slice of a level of mario, and without the
level indicator, detect which level mario is currently in.

## Methods

### Playing the Game
[Just like Sethbling](https://www.youtube.com/watch?v=qv6UVOQ0F44), we are utilizing the power of NEAT algorithms to train our playing model.
NEAT stands for NeuroEvolution of Augmented Topologies, and is a genetic algorithm that continually evolves neural networks. 
This process is derived from evolution in nature where, over the course of centuries and millennia, the best features 
are implemented into the next generation while the worst features are weeded out. The crux of this method is that it 
trains the model by making a plethora of mistakes in order to succeed.

Applying this concept to a level of Mario, the network starts with essentially zero information, 
making random moves until it gets stuck or dies. The player object goes through multiple generations, 
learning where to go and which enemies to avoid. This process can take some time, anywhere from an hour 
to over a day of learning, until the player object successfully completes the level without dying and 
potentially better than human players.

We also plan on implementing a clustering analysis (likely GMM/EM) of the various models we create, to see which ones worked and which ones didn't and why.

### Memorizing the game

To extend this even further, we want the ability to detect which level Mario is in across the entire game without the level indicator. 
We are using classification via Convolutional Neural Networks (CNNs). CNNs are a regularized version of fully connected networks
that use the power of convolutional kernels defined by a width and height passed through a filter to achieve regularization and translation invariance.
The key aspect of CNNs for our project is the translation, or shift, invariance, meaning that certain definined features translated in some way
around are treated as equal, rather than different. CNNs also have a less chance of overfitting the training data, which is an obvious plus.
Commonly used in image and video recognition, we believed this would be a great class of network to use for our object.

Specifically, a successful classification would look like inputting a screen sized image of a Super Mario Bros. level into the network,
without the level indicator, and the output would be the correct classification of the level which the image is from.
There are 32 level classes - 4 levels from 8 worlds.
We are hoping that with CNNs' shift invariance, the model will be able to find key features of each level and recognize them
wherever on the screen, as opposed to the fully connected's method of detection.

We are using 208x240 pixel slices cut out ourselves from the full Super Mario Bros. level maps
from [this website](https://nesmaps.com/maps/SuperMarioBrothers/SuperMarioBrothers.html).
The extended and reduced image size of these level images make for slightly faster computation and
easier acquisition of the actual dataset.

## Desired Outcome

### Playing the Game
As a base goal, we want the computer to be able to complete a level of a mario game with a better time than the
control (the unedited reference script). This will reduce training time significantly to allow for extension. We can likely
accomplish this through multiple means, via increasing information put into the algorithm or
rewarding certain beneficial changes more than others.

Next, we want to introduce an incentive to complete a level faster than a skilled human can. 
We intend to “reinforce” the computer to finish levels faster. We will be satisfied with the 
algorithm when it is able to beat us in finishing the level even after we have practiced for a while.

After successfully training the computer to complete a single level, we hope to implement the same 
algorithm on multiple different levels to assess its adaptability. The computer should be able to 
learn how to play almost any standard level it comes across after it has been trained.  So, after the computer has 
trained on a wide variety of levels, we want it to be able to play any other level with only a little 
difficulty the first time through, just like a skilled human can. This is likely a problem of generalization,
and requires a careful balance.

For the unsupervised analysis, we want to be able to apply our learned information to better program our NEAT algorithm.
This is likely done through identifying which features cluster towards higher scores and what features all levels generate.

### Memorizing the game

The overall goal here is to be able to detect the level input using the CNN.
An accuracy above 90% would be ideal, but we don't exactly know how achievable this is with all levels combined.
Another thing to consider are the underground/secret portions of levels. Some of these are copy/pasted directly from one level
to the next, so we wonder if we could still include them or do we have to purge them from the dataset.

In the end, we also want to be able to determine which levels were the hardest for the model to distinguish so that we may improve the
design in the future.

## Development and Results

### NEAT Algorithm

| ![NEAT Algorithm](img/mario.JPG) |
|:--:|
| *An example of what the NEAT Algorithm looks like in action.* |

At the moment, we have a working NEAT algorithm that is applied to the game Super Mario Bros. Our program runs using the BizHawk emulator, and was heavily inspired by [this great video](https://www.youtube.com/watch?v=qv6UVOQ0F44) by Seth Bling. The NEAT algorithm is a reinforcement learning model, so that means that our model learned by making entirely random moves. The model has rewards based on how far the agent got in the level and punishments based on both the lack of movement and encountering an enemy.

| ![NEAT Algorithm](img/reinforcement.JPG) |
|:--:|
| *The green lines represent positive connections, while the red lines represent negative connections.* |

Each attempt has an associated fitness score based on these factors, and a new attempt is made when the agent either dies or stays still for too long. Each of these attempts at completing the level is considered a different “species” in each generation. After enough attempts are recorded, a new generation is created with the species that had the maximum fitness recorded. 

| ![NEAT Algorithm](img/gen-info.JPG) |
|:--:|
| *The data associated with each generation. The fitness number is the key piece of data for the algorithm.* |

Variations of this species are implemented into this next generation, and after enough generations, the agent will learn what button inputs are necessary to complete the level at specific times. After training the model for over forty-eight hours and sixty generations, the agent was able to successfully complete the first level of Super Mario Bros.

After the initial test of the NEAT Algorithm, we wanted to try to tune hyperparameters to train the agent as efficiently as possible.  After tuning and experimenting, we were able to reduce the original training time by about 20%. The majority of this progress was made by changing how the agent saw gaps in the ground. If Mario was approaching a large jump, we made more explicit instructions for the agent to get through the obstacle rather than wait for the algorithm to stumble across the solution of jumping.

### Level Classifier

To avoid trivial level classification with the level indicator, we grabbed the full maps of each 
level from [this online source](https://nesmaps.com/maps/SuperMarioBrothers/SuperMarioBrothers.html) 
*without* the game HUD. This standardized our image height to 208 pixels, and then the maps were then 
sliced with a 240-pixel-width “sliding window”, with a stride of 4. This size is around the size of the screen
displayed via an emulator. The method produces around 800 images for most levels.
Some level have higher or lower image counts due to level length, but this should not affect the data.

| ![Example Image slice](img/example_slice.png) |
|:--:|
| *An example image slice from the first world.* |

For our images, we know some levels will likely be similar and we want to understand which of the levels the classifier would likely
have a hard time distinguishing. Since the NES only has a specific number of colors available, we decided to count up
these colors for each individual image slice. To do this, we had to take each RGB value from the image, turn the value into byte form,
and count up each unique value. This produced a 37 dimensional vector for each image.

To determine which levels were similar in color to one another, we decided KMeans was the simplest way to do so.
GMM/EM might conform better, but we want hard assignments in terms of color quantity. When we ran this entire dataset 
through the KMeans Elbow Method via scikitlearn, we got the best results at 12 clusters.

| ![Elbow Method sklearn with Kmeans and color quantity dataset](img/elbow_method.png) |
|:--:|
| *The elbow method visualization. Attempted from n_clusters of 4 to 50.* |

The results at 12 clusters showed the parts of levels were similar enough in color quantity to result
in two or more different images from different levels to be labelled under the same cluster. Although this shows
that classifying all the levels will be fairly difficult, it helped us understand which levels were likely to be
mistaken for one another by the model. For example, levels 2-2 and 7-2 are almost identical. In fact, 
the levels are the exact same layout but with more Bloopers (squid enemies), as seen below:

| ![2-2 image slice](img/2-2_slice.png)&nbsp;&nbsp;&nbsp;![7-2 image slice](img/7-2_slice.png) | 
|:--:| 
| *Image slices from 2-2 and 7-2 respectively. The underwater levels are especially quite difficult to tell the difference between. Note the only change is the addition of the Blooper on the left side of the image.* |

We can also take apart a single level and analyze what each cluster looks like. For simplicity let's take the cluster assignments
of each snippet of 1-1 and show the proportions of each assigned cluster.

| ![pie graph of clusters within 1-1](img/1-1_clusterification.png) |
|:--:| 
| *The cluster assignment for all the image slices of 1-1.* |

Now if we take example slices for each, we can show what the general color of each cluster looks like.

| ![blue sky-ish slice of 1-1](img/1-1_cluster_1.png) | ![block ramp from 1-1](img/1-1_cluster_10.png) | 
|:--:| :--:| 
| *Example from Cluster 1. Notice the large amount of blue from the sky, with lesser proportion of the reddish-brown of the ground* | *Example from Cluster 10. Now there is a considerable more proportion of the reddish-brown due to the block ramp, leading to a different cluster assignment.* |

If we do this for all clusters, we can actually see the different 'colors spaces' in each level, and compare each level's pie graph
immediately
(not shown though, because it takes a while and takes up a large amount of space)
 
To further delve into this section, we then created a basic convolutional neural network for the first world using Pytorch, with two convolution layers and three 
fully connected layers. The input is the aforementioned 208x240 image slice, and the output is a 4 dimensional probability vector, 
with each dimension corresponding to the probability that the input is for one of the possible 4 levels in the 
first world. Our optimizer was Adam, and the loss we used was cross entropy, mostly because these we were the ones we were most familiar with.
We split our data into 80% training and 20% testing. After training the model, we ended with a 95% accuracy on the 
test set. With a batch size of 16 and only one epoch, we achieved these results. When we attempted to add more epochs,
the model overfit due to the small number of classes.

| ![Basic CNN network visualization pipeline](img/nn.png) | 
|:--:| 
| *The basic network visualized as a diagram. Relu and batch norm layers are omitted.* |

Pretty awesome, but we knew that classifying 
all 32 levels will be much more difficult. When attempting to 
generate the data for all 32 levels, we actually ran into a memory issue 
where the original dataset could not be loaded with our machines. 
Unfortunately, we had to cut our dataset into half
of what we had. 

This dataset increase also brought us to design our model to be much more
complex. We had to now classify over 4000 208x240 images
into 32 different bins. To do so, we increased both the number of convolutional
and fully connected layers to 4 and heavily increased the channels on the convolutional
layers. These decisions allow for more room to process the image information.


| ![Final CNN network visualization pipeline](img/nn2.png) | 
|:--:| 
| *The final network visualized as a diagram. Relu and batch norm layers are omitted.* |

With the increased dataset sized, we also decided to split the dataset further with a 
80-10-10 of training-validation-test. We kepy the same optimizer (Adam) and loss
(cross entropy). After trying with multiple parameters we promising
results against the validation set.

| Epochs | Batch Size | Learning Rate | Validation Accuracy |
|:-:|:-:|:-:|:-:|
| 10 | 128 | 0.001 | 73.8 |
| 10 | 256 | 0.0005 | 90.14 |
| 10 | 256 | 0.001 | 88.5 |
| 12 | 128 | 0.0005 | 91.02 |
| 12 | 256 | 0.001 | 92.5 |
| 15 | 128 | 0.0005 | 92.74 |
| 15 | 128 | 0.001 | 91.31 |
| 15 | 256 | 0.0005 | 91.81 |
| 18 | 128 | 0.0005 | 91.7 |
| 20 | 128 | 0.001 | 92.74 |
| 20 | 256 | 0.0005 | 88.99 |
| 20 | 256 | 0.001 | 90.97 |


We decided to select the hyperparameters of 15 epochs, batch size of 128, and
learning rate of 0.0005 due to the less stress on runtime and tie for highest validation
accuracy. We ran this again and then tested against the test set, and got a final accuracy of 92%. This is much better than the guess percentage of 3% (1/32).
Our F1-score is not listed as the class distribution is fairly equal and difficult to
analyze with a multi-class set up.

| ![Average loss graph of selected model](img/avgloss.png) | ![Validation accuracy graph of selected model](img/vtacc.png) | 
|:--:| :--:| 
| *The average loss of the selected model of the epochs ran.* | *The validation accuracy for each epoch ran along with the final test accuracy tangent.* |

## Challenges

### NEAT Algorithm

There are several challenges that may arise when attempting 
to apply a NEAT algorithm to some problem. For us, it seems to 
be finding the right hyper-parameters and reward system to make the 
long training times worth our while. New obstacles always prove to 
be a problem, taking the agent hours at a time to learn a way to 
conquer them. This "specialization" of a single obstacle can also be 
detrimental in its ability to beat obstacles that previously caused no problems, 
just making matters worse. 

The biggest problem we have encountered with the NEAT algorithm is gathering sufficient
data to assess the effectiveness of how we tune the hyper-parameters for the model. Since
each model takes about a day and a half of continuous training to run, we cannot gather enough
statistical data to judge how the model responds to our tuning of hyperparameters. With
more computing power, we believe that we would more accurately be able to improve the training
time of the NEAT algorithm through hyper-parameter tuning.

### Level Classifier

Since this game is rather old, the level design and color palette used 
throughout the levels is very similar. This makes classifying each level 
quite a challenge as pixel composition from level to level does not vary 
greatly. There are exceptions, but just from looking at the cluster assignments 
of 2-2 and 7-2 it is evident some levels have exceptional overlap.

| ![cluster with 2-2 slices](img/2-2_clusterification.png)&nbsp;&nbsp;&nbsp;![cluster with 7-2 slices](img/7-2_clusterification.png) | 
|:--:|
|*Cluster assignemnts of levels 2-2 and 7-2. Notice how they were assigned to the same cluster making them nearly impossible to tell apart* |

Another large issue with the level classifier is our personal lack of computing
resources. Both our dataset and our model design our limited to what we are able to run
on our own personal computers, so we had to limit the mentioned two aspects. With more
resources, we would likely be able to improve our performance metrics and 
try new ideas.

The dataset for the image classifier is also very clean compared to a regular
image taken on a monitor. For best realistic results, onoe would have to use a screenshot
of an emulator running Super Mario Bros. and adjust the image size accordingly. 
With photos of a TV screen, this might prove difficult to get accurate results without
training with more varied data.

## Discussion and Future Work

### NEAT Algorithm 

The majority of the work left to be done resides in the neat algorithm. A lot of improvements can be made if 
we expand our knowledge of the LUA scripting language. We felt a bit handicapped trying to produce high level concepts 
with our code while only knowing the bare-minimum. Specifically we would look into optimization techniques in an attempt 
to mitigate the abysmal training times. Another thing that may arise is additional changes to the agent and its perception of the world. 
Since each level is unique, we may have to add in more necessary changes like the gap detection to make our agent 
perform well on certain unencountered obstacles. Overall, we believe great progress was made and this is a wonderful starting point 
for future Machine Learning endeavors.

Furthermore, at a high level, we hope this will entice us to explore more into the generalization of models when using NEAT to play
video games, and possibly explore into training a model to play more complex games like Kirby.

### Level Classifier

Personally, we call the level classifier a successful project. Reaching a 
92% accuracy detection on all the levels in the original Super Mario Bros. is
a cool feat. The analyzation with Kmeans also proved to be a 
valuable insight in determining which levels are similar, and how to 
expect our accuracy to drop based on the design and information space of our model.
Although we were limited by our resources, we made do, and hopefully
we can pick this project up in the future with more resources.

More analyzation could be done with each specific level, determining which levels
were actually misclassified and then iterating on those results could prove promising.
Another thing to consider is gradient visualization. This would let us actually get a 
glance at the individual detection methods for each slice of a level.

Furthermore, we could extend the model to include more than just Super Mario Bros. and possibly "Super Mario Bros.: The Lost Levels" or even "Super Mario Bros. 3".

## References

[Evolving neural networks through augmenting topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)  
KO Stanley, R Miikkulainen - Evolutionary computation, 2002 - MIT Press

[Efficient Exploration In Reinforcement Learning](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.45.2894)  
Sebastian B. Thrun - 1992

[Efficient Reinforcement Learning Through Evolving Neural Network Topologies](http://nn.cs.utexas.edu/?stanley:gecco02b)  
Kenneth O. Stanley and Risto Miikkulainen - 2002

[Dota 2 with Large Scale Deep Reinforcement Learning](https://arxiv.org/abs/1912.06680)  
OpenAI: Christopher Berner and Greg Brockman and Brooke Chan and Vicki Cheung and Przemysław Dębiak and Christy Dennison and David Farhi and Quirin Fischer and Shariq Hashme and Chris Hesse and Rafal Józefowicz and Scott Gray and Catherine Olsson and Jakub Pachocki and Michael Petrov and Henrique Pondé de Oliveira Pinto and Jonathan Raiman and Tim Salimans and Jeremy Schlatter and Jonas Schneider and Szymon Sidor and Ilya Sutskever and Jie Tang and Filip Wolski and Susan Zhang - 2019

[MarI/O - Machine Learning for Video Games](https://www.youtube.com/watch?v=qv6UVOQ0F44)
[Link to reference script](https://pastebin.com/ZZmSNaHX)
Sethbling, [check out his awesome videos](https://www.youtube.com/channel/UC8aG3LDTDwNR1UQhSn9uVrw)

[Parallel distributed processing model with local
space-invariant interconnections and
its optical architecture](https://drive.google.com/file/d/0B65v6Wo67Tk5ODRzZmhSR29VeDg/view)
Wei Zhang, Kazuyoshi Itoh, Jun Tanida, and Yoshiki Ichioka
