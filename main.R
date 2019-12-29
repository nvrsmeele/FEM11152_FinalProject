#----
# Setup workspace
#----

# Load libraries
library(readr)
library(dplyr)
library(plyr)
library(lubridate)
library(data.table)
library(ggplot2)
library(RColorBrewer)
library(caTools)
library(text2vec)
library(textmineR)
library(caret)
library(naivebayes)
library(randomForest)
library(iml)
library(readr)

# Load full dataset
yelp_reviews <- read_csv("yelp_academic_dataset_review.csv")
yelp_business <- read_csv("yelp_academic_dataset_business.csv")

#----
# 1. Data preprocessing
#----

##----
## 1.1 Dataset "yelp_business"

# Remove unnecessary columns
business <- yelp_business[, c(-2:-10, -12:-22, -24:-37)]
business <- business[, c(-5:-26, -29:-31, -33:-71)]

# Rename columns
business$business_name <- business$name
business <- business[,-3]

business$avg_stars <- business$stars
business <- business[,-6]

# Remove unnecessary columns
business <- business[, -5:-7]

# Split character value in "Categories" variable
business$categories <- strsplit(business$categories, split = ";")

##----
## 1.2 Dataset "yelp_reviews"

# Remove unnecessary columns
yelp_reviews <- yelp_reviews[, c(-1:-2, -4, -6, -9:-10)]

##----
## 1.3 Create new dataframe "reviews" as subset of selected restaurants

# Merge dfs "business" and "yelp_reviews"
reviews_full <- inner_join(yelp_reviews, business)

# Create df "reviews" as subset of all "Diner" restaurants
reviews_subset <- reviews_full[which(grepl("Diners", reviews_full$categories)),]
reviews <- subset(reviews_subset, subset = open == TRUE)  # remove all closed diners
reviews <- subset(reviews, subset = review_count > 20) # remove bias

# Split date column into seperate columns
reviews <- reviews %>%
              mutate(year = year(date),
                     month = month(date),
                     day = day(date))

# Visualize frequency over years
ggplot(reviews, aes(x = year))+
  geom_bar(col = "red", fill = "blue", alpha = .5)+
  scale_x_continuous(breaks = seq(2004, 2014, 1))+
  labs(title = "Number of reviews over the years for all local diners")

##----
## 1.4 Dataset "reviews_subset" and create workable dataset as "reviews_final"

# Check data structure
str(reviews)

# (re-)Factor dependent variable "Star sentiment"
reviews$stars <- factor(reviews$stars, ordered = TRUE)
reviews$sentiment <- mapvalues(reviews$stars, from = c("1", "2", "3","4", "5"),
                           to = c("negative", "negative", "average","positive", "positive"))

# Visualize review distribution over sentiment categories
ggplot(reviews, aes(x = sentiment))+
  geom_bar(col = "red", fill = "blue", alpha = .5)+
  labs(title = "Number of reviews per sentiment class", x = "Sentiment class")

# Add "Document IDs" to dataframe
reviews$doc_id <- rownames(reviews)

# Clean work environment to free memory
rm(business, reviews_full, yelp_business, yelp_reviews, reviews_subset)

##----
## 1.5 Text data preprocessing (dataset "reviews_subset")

# Function for text cleaning to reduce noise
text_clean <- function(x){
  require("tm")
  x <- stripWhitespace(x) # remove all extra whitespaces
  x <- iconv(x, "latin1", "ASCII", sub="")  # keep only ASCII characters
  x <- tolower(x) # set all letters to lower-cases
  x <- removePunctuation(x) # remove all punctuations
  x <- removeWords(x, stopwords("english")) # remove all stopwords
  x <- removeNumbers(x) # remove all numbers
  x <- stemDocument(x) # apply stemming to entire corpus
  return(x)
}

# Word iterator initialization to create a vocabulary
word_token <- itoken(reviews$text,
              preprocessor = text_clean,
              tokenizer = word_tokenizer,
              ids = reviews$doc_id)

# Create unigram vocabulary
vocab <- create_vocabulary(word_token, ngram = c(1L, 1L))

# Prune vocabulary by removing words that occur less than 1% of all documents
pruned_vocab <- prune_vocabulary(vocab, doc_proportion_min = 0.01)

# Transform list of tokens into vector space
vectorizer <- vocab_vectorizer(pruned_vocab)

# Create Document-term matrix
text_dtm <- create_dtm(word_token, vectorizer)

#----
# 2. Latent Dirichlet Allocation (LDA) modeling
#----

##----
## 2.1 Find number of K-topics by default settings

# Initialize empty numeric vector
lda_coh <- numeric(20)

# Fit LDA model 20 times by iterating over topics and default alpha/beta
for (i in 1:20) {
  # Set random seed
  set.seed(12345)
  
  # Train LDA model
  temp <- FitLdaModel(dtm = text_dtm, k = i, iterations = 1000, calc_coherence = TRUE,
                     alpha = 0.1, beta = 0.05, burnin = 100)
  
  # Store coherence measure for each iteration
  lda_coh[i] <- mean(temp$coherence)
  
  # Print checkmark for each iteration
  print(i)
}

# Visualize coherence measure over 20 topics
ntopics <- seq(1, 20, by = 1) # declare vector with number of topics
lda_run1 <- as.data.frame(cbind(lda_coh, ntopics)) # create df with all info

ggplot(lda_run1, aes(x = ntopics, y = lda_coh))+
  geom_line(col = "black")+
  geom_point(col = "black")+
  labs(title = "Coherence measure per topic", x = "number of topics",
       y = "Coherence measure")+
  geom_vline(xintercept = 17, linetype = "dotted", color = "red", size = 1)

##----
## 2.2 Tune for alpha and beta

# Create grid search with alpha and beta values
hyper_grid <- expand.grid(
  k = 17,
  alpha = c(0.1, 1.5, 2.94, 4),
  beta = c(0.05, 0.23, 0.5, 1),
  coherence = 0
)

# Run LDA model iteratively over hyper_grid
for (i in 1:nrow(hyper_grid)) {
  # Set random seed
  set.seed(123)
  
  # Train LDA model
  lda_tune <- FitLdaModel(dtm = text_dtm, k = hyper_grid$k[i], iterations = 1000,
                          burnin = 100, alpha = hyper_grid$alpha[i],
                          beta = hyper_grid$beta[i], calc_coherence = TRUE)
  
  # Store coherence measure for each iteration
  hyper_grid$coherence[i] <- mean(lda_tune$coherence)
  
  # Print checkmark for each iteration
  print(i)
}

# Obtain LDA model with highest coherence measure with tuned parameters
best_par <- hyper_grid[which(hyper_grid$coherence == max(hyper_grid$coherence)),]
k_opt <- best_par[1,1] # best k-topics parameter
alpha_opt <- best_par[1,2] # best alpha parameter
beta_opt <- best_par[1,3] # best beta parameter

# Visualize coherence measure over alpha/beta combinations
hyper_grid1 <- hyper_grid # duplicate 'hyper_grid' for manipulation
hyper_grid1$beta <- factor(hyper_grid1$beta) # factorise 'beta'

ggplot(hyper_grid1, aes(x = alpha, y = coherence, fill = beta))+
  geom_bar(position = "dodge", stat = "identity")+
  scale_x_continuous(breaks = c(0.1, 1.5, 2.94, 4))+
  scale_fill_brewer(palette="Spectral")+
  labs(title = "Coherence measure for each alpha/beta combination",
       y = "Coherence measure")

# Fit best LDA model with optimal parameters
set.seed(1235)
lda_best <- FitLdaModel(text_dtm, k = k_opt, iterations = 1000, alpha = alpha_opt,
                        beta = beta_opt, calc_coherence = TRUE, calc_r2 = TRUE,
                        burnin = 100, calc_likelihood = TRUE)

##----
## 2.3 Determine latent topics and convert each to feature vector

# Visualize top-10 terms per topic
topTerms <- GetTopTerms(phi = lda_best$phi, M = 10)

# Convert topics to feature vectors
topic_features <- as.data.frame(lda_best$theta)

# Re-name topics and merge overlapping topics in feature vectors
topic_features$breakfast_menu <- topic_features$t_1 + topic_features$t_4
topic_features$service <- topic_features$t_2 + topic_features$t_6
topic_features$dinner_menu <- topic_features$t_3 + topic_features$t_5
topic_features$opening_hours <- topic_features$t_7
topic_features$overall_quality <- topic_features$t_8
topic_features$food_quality <- topic_features$t_9
topic_features$lunch_menu <- topic_features$t_10
topic_features$price_quality <- topic_features$t_11
topic_features$food_offer <- topic_features$t_12
topic_features$dessert_menu <- topic_features$t_13
topic_features$visit_freq <- topic_features$t_14
topic_features$location <- topic_features$t_15
topic_features$ambience <- topic_features$t_16
topic_features$guest_exp <- topic_features$t_17

# Remove all original vectors from df
topic_features <- topic_features[,-1:-17]

# Add sentiment labels to df
topic_features$sentiment <- reviews$sentiment

#----
# 3. Preparation for classification modeling
#----

##----
## 3.1 Create training & test set

# Create training/testing set by random selection
set.seed(111)
samp <- sample.split(topic_features$sentiment, SplitRatio = 2/3)
train <- subset(topic_features, samp == TRUE) # declare training set
test <- subset(topic_features, samp == FALSE) # declare test set

ytrain <- train[,15] # declare training response variable
xtrain <- train[,1:14] # declare training predictor variables

ytest <- test[,15] # declare test response variable
xtest <- test[,1:14] # declare test predictor variables

##----
## 3.2 Solve for imbalanced dataset

# Downsample the training set to reduce bias towards "4-5" class
trainDown <- downSample(xtrain, ytrain, yname = "sentiment")
xtrainDown <- trainDown[,1:14]
ytrainDown <- trainDown[,15]

#----
# 4. Classification modeling
#----

##----
## 4.1 Naive Bayes classifier

# Step 1: Create NB classifier
nb_model <- naive_bayes(as.matrix(xtrainDown), ytrainDown, usekernel = TRUE)

# Step 2: Generate predictions of the NB classifier
pred <- predict(nb_model, data = test, type = "class")

# Step 3: Create confusion matrix of NB's prediction performance
nb_cfm <- confusionMatrix(data = pred, reference = ytrainDown)

##----
## 4.2 Random Forest classifier

# Step 1: Run initial Random Forest model
set.seed(200)
rf_model <- randomForest(xtrainDown, ytrainDown, mtry = 4, replace = TRUE,
                         importance = TRUE, ntree = 1500, do.trace = TRUE,
                         control = rpart.control(minsplit = 2, cp = 0))

# Step 2: Find OOB error convergence to determine 'best' ntree
err_best.tree <- as.data.frame(rf_model$err.rate)
err_best.tree$ntree <- as.integer(rownames(err_best.tree))

ggplot(err_best.tree, aes(x = ntree, y = OOB))+
  geom_line(col = "black")+
  labs(title = "Random Forest (B) bootstrap samples", x = "number of bootstrap samples",
       y = "OOB error")+
  geom_vline(xintercept = 1000, linetype = "dotted", color = "red", size = 1)

# Step 3: Find 'best' mtry
mtry_err <- vector(length = 14) # initialize empty vector

# Run 7-times random forest model and obtain OOB estimated error per interation
for(i in 1:14){
  set.seed(205)
  temp <- randomForest(xtrainDown, ytrainDown, mtry = i, replace = TRUE,
                       importance = TRUE, ntree = 1000, do.trace = TRUE,
                       control = rpart.control(minsplit = 2, cp = 0))
  mtry_err[i] <- temp$err.rate[1000,1]
}

# Store all 7 OOB estimated errors in matrix
mtry <- c(1:14)
mtry_mx <- data.frame(cbind(mtry, mtry_err))

# Visualize OOB error per mtry
ggplot(mtry_mx, aes(x = mtry, y = mtry_err))+
  geom_line(col = "black")+
  geom_point()+
  labs(title = "Random Forest (m) predictor parameter", x = "number of (m) predictors",
       y = "OOB error")+
  geom_vline(xintercept = 4, linetype = "dotted", color = "red", size = 1)

# Step 4: Run final Random Forest model
set.seed(215)
rf_best <- randomForest(xtrainDown, ytrainDown, mtry = 4, replace = TRUE,
                        importance = TRUE, ntree = 1000, xtest = xtest, ytest = ytest,
                        do.trace = TRUE, control = rpart.control(minsplit = 2, cp = 0))

# Step 5: Performance evaluation by OOB estimated error/accuracy
rf_ooberr <- round(rf_best$err.rate[1000,1], digits = 4)
rf_testerr <- round(rf_best$test[["err.rate"]][1000,1], digits = 4)
rf_oobacc <- 1 - rf_ooberr
rf_testacc <- 1 - rf_testerr
rf_testpf <- rbind(rf_testerr, rf_testacc)
rf_oobpf <- rbind(rf_ooberr, rf_oobacc)
rf_perform <- as.matrix(cbind(rf_oobpf, rf_testpf))
rownames(rf_perform) <- c("Error", "Accuracy")

# Step 6: Store confusion matrix
rf_cfm <- rf_best$confusion

#----
# 5. Best classifier interpretation
#----

##----
## 5.1 Premutation Feature Importance

# Step 1: Declaration of the prediction function
pred_nb <- function(model, newdata){
  newdata <- as.matrix(newdata)
  predict(model, newdata, method = "class")
}

# Step 2: Create prediction container for each sentiment class
pred_negative <- Predictor$new(nb_model, data = xtest, type = "class",
                           y = ytest == "negative", class = 1, predict.fun = pred_nb)

pred_average <- Predictor$new(nb_model, data = xtest, type = "class",
                              y = ytest == "average", class = 2, predict.fun = pred_nb)

pred_positive <- Predictor$new(nb_model, data = xtest, type = "class",
                               y = ytest == "positive", class = 3, predict.fun = pred_nb)

# Step 3: Compute feature importance for each sentiment class
set.seed(999) # negative sentiment class
fi_negative <- FeatureImp$new(pred_negative, loss = 'ce', compare = 'ratio',
                              n.repetitions = 20)

set.seed(999) # average sentiment class
fi_average <- FeatureImp$new(pred_average, loss = 'ce', compare = 'ratio',
                              n.repetitions = 20)

set.seed(999) # positive sentiment class
fi_positive<- FeatureImp$new(pred_positive, loss = 'ce', compare = 'ratio',
                             n.repetitions = 20)

# Step 4: Plot the PFI for each class
fi_negative$plot()+ # negative sentiment class
  labs(title = "Negative sentiment class")
  
fi_average$plot()+ # average sentiment class
  labs(title = "Average sentiment class")

fi_positive$plot()+ # positive sentiment class
  labs(title = "Positive sentiment class")

# Step 5: Print PFI results for each class
fi_negative$results # negative sentiment class
fi_average$results # average sentiment class
fi_positive$results # positive sentiment class



