## FEM11152 Final Project (autumn 2019)

This project uses a semantic and classification approach for Yelp reviews to identify which restaurant features are the main drivers for negative, average and positive star-ratings on the platform (see [paper](https://github.com/nvrsmeele/FEM11152_FinalProject/blob/master/paper/FinalPaper_MLseminar.pdf) and [code](https://github.com/nvrsmeele/FEM11152_FinalProject/blob/master/code/main.R)).

### Project background
Yelp is a platform and information source that helps customers to plan a night out and locate local businesses based on social networking functionalities such as reviews and star-ratings. However, reviews are not only valuable to customers but are providing valuable insights for restaurant owners as well. Local restaurants can identify what elements their customers like or what needs to be improved in their service and food offering to keep their customers satisfied.

In general, star-ratings provide an overall indication of whether customers were satisfied and do not indicate what features stimulated them to give the specific rating. This is a challenge for business owners since reviews are often technologically poor; they often have no choice but to browse through massive amounts of text to find interesting information.

The dataset used for this project is part of the [Yelp's Dataset Challenge](https://www.yelp.com/dataset/challenge).

### Models
To tackle this challenge, the Latent Dirichlet Allocation (LDA) model is used to uncover the latent topics in each text document. Based on the obtained latent topics, the Naive Bayes (NB) classifier is used as a base model and compared to a Random Forest (RF) model to predict the sentiment of each document. The last step is to identify which latent topics drives the predicted sentiment. For this task, the model-agnostic technique called Permutation Feature Importance (PFI) is used.

### Course result
This coursework received a 9.3 out of 10 as grade; the small missing piece was that I should have explained the methods a bit more in-depth from a technical point of view, including the Markov Chain Monte Carlo (MCMC) algorithm Gibbs sampling to estimate the Dirichlet prior.
