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
library(caTools)
library(text2vec)
library(textmineR)
#library(tm)
#library(textstem)
#library(lexicon)
library(naivebayes)
library(caret)
library(randomForest)
library(iml)
#library(rpart)
#library(readr)
#library(stringr)
#library(car)
library(keras)
library(tensorflow)
library(reticulate)

reticulate::use_python("/opt/anaconda3/envs/r-base/bin")

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
  geom_bar(col = "red", fill = "green", alpha = .2)+
  scale_x_continuous(breaks = seq(2004, 2014, 1))

# Subet all restaurant reviews from 2014
#reviews <- reviews[reviews$date %between% c("2012-01-01", "2012-12-31"),]

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
  geom_bar(col = "red", fill = "green", alpha = .2)

# Add "Document IDs" to dataframe
reviews$doc_id <- rownames(reviews)

# Clean work environment to free memory
rm(business, reviews_full, yelp_business, yelp_reviews, reviews_subset)

##----
## 1.5 Text data preprocessing (dataset "reviews_subset")

# Function for text cleaning to reduce noise
text_clean <- function(x){
  require("tm")
  x <- stripWhitespace(x)
  x <- iconv(x, "latin1", "ASCII", sub="")  # keep only ASCII characters
  x <- tolower(x)
  x <- removePunctuation(x)
  x <- removeWords(x, stopwords("english"))
  x <- removeNumbers(x)
  x <- stemDocument(x)
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
plot(ntopics, lda_coh, type = "l") # plot coherence measure for all topics

##----
## 2.2 Tune for alpha and beta

# Create grid search with alpha and beta values
hyper_grid <- expand.grid(
  k = 17,
  alpha = c(0.1, 1.5, 2.94, 4),
  beta = c(0.1, 0.23, 0.5, 1, 1.5),
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

# Fit best LDA model with optimal parameters
set.seed(1235)
lda_best <- FitLdaModel(text_dtm, k = k_opt, iterations = 1000, alpha = alpha_opt,
                        beta = beta_opt, calc_coherence = TRUE, calc_r2 = TRUE,
                        burnin = 100, calc_likelihood = TRUE)

folds <- 5
splitfolds <- sample(1:folds, nrow(text_dtm), replace = TRUE)
train_set <- text_dtm[splitfolds != 1 , ]
valid_set <- text_dtm[splitfolds == 1, ]

test <- predict(lda_best, newdata = valid_set)

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
samp <- sample.split(topic_features$stars, SplitRatio = 2/3)
train <- subset(topic_features, samp == TRUE) # declare training set
test <- subset(topic_features, samp == FALSE) # declare test set

ytrain <- train[,8] # declare training response variable
xtrain <- train[,1:7] # declare training predictor variables

ytest <- test[,8] # declare test response variable
xtest <- test[,1:7] # declare test predictor variables

##----
## 3.2 Solve for imbalanced dataset

# Downsample the training set to reduce bias towards "4-5" class
trainDown <- downSample(xtrain, ytrain, yname = "stars")
xtrainDown <- trainDown[,1:7]
ytrainDown <- trainDown[,8]

#----
# 4. Classification modeling
#----

##----
## 4.1 Naive Bayes classifier

# Step 1: Create NB classifier
nb_model <- multinomial_naive_bayes(as.matrix(xtrainDown), ytrainDown, laplace = 0.5)

# Step 2: Generate predictions of the NB classifier
pred <- predict(nb_model, data = test, type = "class")

# Step 3: Create confusion matrix of NB's prediction performance
nb_cfm <- confusionMatrix(data = pred, reference = ytrainDown)

##----
## 4.2 Random Forest classifier

# Step 1: Run initial Random Forest model
set.seed(200)
rf_model <- randomForest(xtrainDown, ytrainDown, mtry = 3, replace = TRUE,
                         importance = TRUE, ntree = 1500, do.trace = TRUE,
                         control = rpart.control(minsplit = 2, cp = 0))

# Step 2: Find OOB error convergence to determine 'best' ntree
plot(rf_model$err.rate[,1], type="l", xlab = "Number of bootstrap samples",
     ylab = "OOB error", main = "Random Forest classifier", las = 1)
abline(v = 1000, col = "red", lty = 3)

# Step 3: Find 'best' mtry
mtry_err <- vector(length = 7) # initialize empty vector

# Run 7-times random forest model and obtain OOB estimated error per interation
for(i in 1:7){
  set.seed(205)
  temp <- randomForest(xtrainDown, ytrainDown, mtry = i, replace = TRUE,
                       importance = TRUE, ntree = 1000, do.trace = TRUE,
                       control = rpart.control(minsplit = 2, cp = 0))
  mtry_err[i] <- temp$err.rate[1000,1]
}

# Store all 7 OOB estimated errors in matrix
mtry <- c(1:7)
mtry_mx <- data.frame(cbind(mtry, mtry_err))

# Visualize OOB error per mtry
plot(mtry_mx$mtry, mtry_mx$mtry_err, type = "b", pch=19,
     xlab = "Number of predictors",
     ylab = "OOB error", main = "Random Forest predictor parameter")
abline(v = 2, col = "red", lty = 3)

# Step 4: Run final Random Forest model
set.seed(215)
rf_best <- randomForest(xtrainDown, ytrainDown, mtry = 2, replace = TRUE,
                        importance = TRUE, ntree = 1000, xtest = xtest, ytest = ytest,
                        do.trace = TRUE, control = rpart.control(minsplit = 2, cp = 0))

rf_best$importance

# Step 5: Performance evaluation by OOB estimated error/accuracy
rf_ooberr <- round(rf_best$err.rate[1000,1], digits = 4)
rf_testerr <- round(rf_best$test[["err.rate"]][1000,1], digits = 4)
rf_oobacc <- 1 - rf_ooberr
rf_testacc <- 1 - rf_testerr
rf_testpf <- rbind(rf_testerr, rf_testacc)
rf_oobpf <- rbind(rf_ooberr, rf_oobacc)
rf_perform <- as.matrix(cbind(rf_oobpf, rf_testpf))
rownames(rf_perform) <- c("Error", "Accuracy")


#----
# 5. Best classifier interpretation
#----

##----
## 5.1 Premutation Feature Importance

# Prediction function
pred_nb <- function(model, newdata){
  newdata <- as.matrix(newdata)
  predict(model, newdata, method = "class")
}

# Multinomial
predictor <- Predictor$new(nb_model, data = xtest, type = "class",
                           y = ytest == "3", class = 2, predict.fun = pred_nb)

imp <- FeatureImp$new(predictor, loss = 'ce', compare = 'ratio', n.repetitions = 20)

imp$plot()











#----
#LDA MODELLING topicmodels
#----
library(topicmodels)

log_lik <- numeric(20)
perplexity <- numeric(20)

for(i in 2:20){
  mod <- topicmodels::LDA(unistem_textDTM, k = i, method = "Gibbs",
         control = list(alpha = 0.5, iter = 500, seed = 12345, thin = 1, verbose = 1))
    
  log_lik[i] <- topicmodels::logLik(mod)
  perplexity[i] <- topicmodels::perplexity(mod, unistem_textDTM)
}

mod_ok <- topicmodels::LDA(unistem_textDTM, k = 2, method = "Gibbs",
                        control = list(alpha = 0.5, iter = 1000, seed = 12345, thin = 1, verbose = 1))


x <- seq(1, 20, by = 1)

plot(x, perplexity, type = "l")
#----


# Create a VCorpus
text <- VCorpus(VectorSource(reviews$text))

# Declare Lemmatization Dictionary
lemma <- function(x) lemmatize_strings(x, dictionary = lexicon::hash_internet_slang)

# Corpus text cleaning
text <- tm_map(text, stripWhitespace)
text <- tm_map(text, content_transformer(tolower))
text <- tm_map(text, removeWords, stopwords("english"))
text <- tm_map(text, removePunctuation)
text <- tm_map(text, removeNumbers)
text_lemm <- tm_map(text, content_transformer(lemma))
text_stem <- tm_map(text, stemDocument)

##----
## 1.5 Ngram modeling based on tf-idf, and remove sparse terms

# Unigram model based on Stemming (base model)
unistem_textDTM <- DocumentTermMatrix(text_stem,
                   control = list(weighting = function(x) weightTf(x)))

sparse <- removeSparseTerms(unistem_textDTM, 0.98) # remove sparse terms
unistem_textDTM <- as.data.frame(as.matrix(sparse)) # convert dtm to matrix
unistem_textDTM$stars <- reviews$stars # add dependent variable to matrix

# Unigram model based on Lemmatization
unilemm_textDTM <- DocumentTermMatrix(text_lemm,
                   control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))

sparse <- removeSparseTerms(unilemm_textDTM, 0.99) # remove sparse terms
unilemm_textDTM <- as.data.frame(as.matrix(sparse)) # convert dtm to matrix
unilemm_textDTM$stars <- reviews$stars # add dependent variable to matrix

match("stars", names(unilemm_textDTM)) # check index of dependent variable

# Bigram model based on Stemming
#----
# Customer function for bigram
bitoken <-  function(x){
  unlist(lapply(ngrams(words(x), 2), paste, collapse = " "), use.names = FALSE)
}

bistem_textDTM <- DocumentTermMatrix(text_stem,
                  control = list(tokenize = bitoken,
                  weighting = function(x) weightTfIdf(x, normalize = TRUE)))

sparse <- removeSparseTerms(bistem_textDTM, 0.99) # remove sparse terms
bistem_textDTM <- as.data.frame(as.matrix(sparse)) # convert dtm to matrix
bistem_textDTM$stars <- reviews_subset$stars # add dependent variable to matrix
rownames(bistem_textDTM) <- rownames(reviews_subset) # add document ID's to rows

#----
# Test with tidytext
#----
# Bigram model based on Stemming
text_tidy <- tidy(text_stem)
text_tidy$id <- as.integer(text_tidy$id)

## Create DocumentTermMatrix for Bigram model based on Stemming
bistem_textDTM <- text_tidy %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  group_by(id) %>%
  dplyr::count(bigram) %>%
  bind_tf_idf(bigram, id, n) %>%
  cast_dtm(id, bigram, tf_idf)

sparse <- removeSparseTerms(bistem_textDTM, 0.99) # remove sparse terms
bistem_textDTM <- as.data.frame(as.matrix(sparse)) # convert dtm to matrix
bistem_textDTM$stars <- reviews_subset$stars # add dependent variable to matrix
#----

#----
# Test with text2vec
#----
library(text2vec)

reviews_subset$docID <- rownames(reviews_subset)

# define preprocessing function and tokenization function
prep_fun <- tolower
tok_fun <- word_tokenizer

it <- tr_te %$%
  str_to_lower(txt) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  tokenize_word_stems(language = "russian") %>% 
  itoken()

it_train <- itoken(reviews_subset$text, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun, 
                   ids = reviews_subset$docID, 
                   progressbar = TRUE)

vocab <- create_vocabulary(it_train)
vocab$term <- stemDocument(vocab$term)

vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)
#----

#----
# 2. Model building preparation
#----

##----
## Split unistem_textDTM randomly for training/test
set.seed(111)
samp <- sample.split(unistem_textDTM$stars, SplitRatio = 2/3)
train.uni <- subset(unistem_textDTM, samp == TRUE) # declare training set
test.uni <- subset(unistem_textDTM, samp == FALSE) # declare test set

ytrain.uni <- train.uni[,504] # declare training response variable
xtrain.uni <- train.uni[,1:503] # declare training predictor variables

##----
## Split unilemm_textDTM randomly for training/text
set.seed(112)
samp <- sample.split(unilemm_textDTM$stars, SplitRatio = 2/3)
train.lemm <- subset(unilemm_textDTM, samp == TRUE)
test.lemm <- subset(unilemm_textDTM, samp == FALSE)

ytrain.lemm <- train.lemm[, 740]
xtrain.lemm <- train.lemm[, c(1:739, 741:898)]

#----
# DownSample
#----
trainDown <- downSample(xtrain.uni, ytrain.uni, yname = "stars")
xtrainDown.uni <- trainDown[,1:503]
ytrainDown.uni <- trainDown[,504]

ytest.uni <- test.uni[,504] # declare test response variable
xtest.uni <- test.uni[,1:503] # declare test predictor variables

## Split bistem_textDTM randomly for training/test
set.seed(150)
samp <- sample.split(bistem_textDTM$stars, SplitRatio = 2/3)
train.bi <- subset(bistem_textDTM, samp == TRUE) # declare training set
test.bi <- subset(bistem_textDTM, samp == FALSE) # declare test set

ytrain.bi <- train.bi[,105] # declare training response variable
xtrain.bi <- train.bi[,1:104] # declare training predictor variables

ytest.bi <- test.bi[,105] # declare test response variable
xtest.bi <- test.bi[,1:104] # declare test predictor variables

##----


#----
# 3. Model building
#----

##----
## multi-class Naive Bayes Classifier (base model)
library(naivebayes)

# Step1: Create NB classifier
nb_model <- multinomial_naive_bayes(as.matrix(xtrainDown), ytrainDown, laplace = 0.5)
nb_model

# Step 2: Generate predictions of the NB classifier
pred <- predict(nb_model, data = test, method = "class")

# Step 3: Create confusion matrix of NB's prediction performance
nb_cfm <- confusionMatrix(data = pred, reference = ytrainDown)

# Step 4: Feature importance
nb_coef <- coef(nb_model)

#Test with VIP/LIME by using Caret
#----

trcontrol <- trainControl(method="none")

nb_model <- caret::train(xtrainDown, ytrainDown,
                  method = "naive_bayes",
                  tuneGrid = data.frame(laplace = 0.5,
                                        usekernel = FALSE,
                                        adjust = FALSE),
                  trControl = trcontrol)

rf_model <- caret::train(xtrainDown, ytrainDown,
                    method='rf', 
                    trControl=trcontrol)  # to get probs along with classifications

svm_model <- caret::train(xtrainDown, ytrainDown,
                    method='svmLinear2',
                    trControl=trcontrol,
                    probability = TRUE)

library(vip)


vip(lime::as.classifier(nb_model), num_features = 5, bar = TRUE)

library(iml)

pred_nb <- function(model, newdata){
  newdata <- as.matrix(newdata[,1:11])
  predict(model, newdata, method = "class")
}

# Feature importance

predictor <- Predictor$new(nb_model, data = xtest, type = "prob",
                           y = ytest)

imp <- FeatureImp$new(predictor, loss = 'ce', compare = 'ratio', n.repetitions = 20)

imp$plot()
imp$results

# LIME
library(lime)
fff<-function(x){as.matrix(x)}

explainer <- lime(x = xtrainDown, model = as_classifier(nb_model), preprocess=fff)
explainer <- lime(x = xtrainDown, model = nb_model, preprocess=fff)

explanation <- lime::explain (
  xtrainDown[10:11,], # Select Jack and Rose
  explainer    = explainer, 
  n_labels     = 1, # Explaining a single class
  n_features   = 5, # Returns top-4 features critical to each case
  kernel_width = 0.5) # Select kernel width

plot_features (explanation) +
  labs (title = "LIME: Feature Importance Visualization")

#----

##----
## SVM Classifier
library(e1071)
library(radiant.data)

svm_model <- svm(x = xtrainDown, y = ytrainDown,
                 type = "nu-classification", kernel = "polynomial")

pred <- predict(svm_model, data = test, method = "class")

svm_cfm <- confusionMatrix(data = pred, reference = ytrainDown)

W <- t(svm_model$coefs) %*% svm_model$SV
weights.df <- as.data.frame(t(W), row.names = NULL)
weights.df <- weights.df %>% rownames_to_column("word")
names(weights.df)[names(weights) == "V1"] <- "w"

weights.df %>% group_by(V1 < 0) %>%
  top_n(20, abs(V1)) %>%
  ungroup() %>%
  mutate(word = reorder(word, V1)) %>%
  ggplot(aes(word, V1, fill = V1 < 0 )) +
  geom_col(show.legend = F) +
  coord_flip() + 
  ylab("w") +
  scale_fill_manual(values = c("orange", "blue"))

##----
## Random Forest Classifier

# Step 1: Run initial Random Forest model
set.seed(200)
rf_model <- randomForest(xtrainDown, ytrainDown, mtry = 3, replace = TRUE,
                         importance = TRUE, ntree = 75, do.trace = TRUE,
                         control = rpart.control(minsplit = 2, cp = 0))

pred <- predict(rf_model, data = test, type = "class")
rf_cfm <- confusionMatrix(data = pred, reference = ytrainDown)
rf_model$importance

# Step 2: Find OOB error convergence to determine 'best' ntree
plot(rf_model$err.rate[,1], type="l", xlab = "Number of bootstrap samples",
     ylab = "OOB error", main = "Random Forest classifier", las = 1)
abline(v = 500, col = "red", lty = 3)

# Step 3: Find 'best' mtry
mtry_err <- vector(length = 12) # initialize empty vector

# Run 12-times random forest model and obtain OOB estimated error per interation
for(i in 1:12){
  set.seed(205)
  temp <- randomForest(Revenue ~ ., data = train, mtry = i, replace = TRUE,
                       importance = TRUE, ntree=500,
                       control = rpart.control(minsplit = 2, cp = 0))
  mtry_err[i] <- temp$err.rate[500,1]
}

# Store all 12 OOB estimated errors in matrix
mtry <- c(1:12)
mtry_mx <- data.frame(cbind(mtry, mtry_err))

# Visualize OOB error per mtry
plot(mtry_mx$mtry, mtry_mx$mtry_err, type = "b", pch=19,
     xlab = "Number of predictors",
     ylab = "OOB error", main = "Random Forest predictor parameter")
abline(v = 5, col = "red", lty = 3)

# Step 4: Run final Random Forest model
set.seed(215)
rf_best <- randomForest(Revenue ~ ., data = train, mtry = 4, replace = TRUE,
                        importance = TRUE, ntree=500, xtest = xtest, ytest = ytest,
                        control = rpart.control(minsplit = 2, cp = 0))

# Step 5: Performance evaluation by OOB estimated error/accuracy
rf_ooberr <- round(rf_best$err.rate[500,1], digits = 4)
rf_testerr <- round(rf_best$test[["err.rate"]][500,1], digits = 4)


##----
## XGBoost Classifier

xgbtrain <- xgb.DMatrix(data = as.matrix(xtrain), label = as.numeric(ytrain)-1)

numberOfClasses <- length(unique(ytrain))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 10

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = xgbtrain, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = ytrain)

OOF_prediction$label <- mapvalues(OOF_prediction$label, from = c("1-2", "3", "4-5"), to = c("1", "2", "3"))

confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")



##----
## Neural Network Classifier

library(keras)
library(tensorflow)
library(reticulate)

reticulate::use_python("/opt/anaconda3/envs/r-base/bin")

xtrain.norm <- normalize(as.matrix(xtrain))
ytrain.bin <- to_categorical(as.numeric(ytrain)-1, 3)
ytest.bin <- to_categorical(as.numeric(ytest)-1, 3)

model <- keras_model_sequential() %>% 
  layer_dense(units = 6, activation = "relu", input_shape = c(7)) %>%
  layer_dense(units = 4, activation = "relu") %>%
  layer_dense(units = 3, activation = "softmax")

model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'categorical_crossentropy',
  metrics = list('accuracy')
)

trained_model <- model %>% fit(
  as.matrix(xtrain),
  ytrain.bin,
  epochs = 50,
  batch_size = 32,
  validation_split = 0.2,
)

model %>% evaluate(xtest, ytest.bin)



#----
# Data Exploration
# source: https://www.kaggle.com/omkarsabnis/sentiment-analysis-on-the-yelp-reviews-dataset
#----

# Explore dataset
str(reviews)

# word length of the review
# break up the strings in each row by " "
temp <- strsplit(reviews$text, split=" ")

# count the number of words as the length of the vectors
reviews$length <- sapply(temp, length)

ggplot(reviews, aes(x = length)) +
  geom_histogram(bins = 50) +
  facet_grid(stars~.)

# Mean Value of the Vote columns
reviews %>%
  group_by(stars) %>%
  summarize(cool = mean(votes.cool), useful = mean(votes.useful),
            funny = mean(votes.funny), length = mean(length))

# Correlation between the voting columns
cor_mat <- reviews[, c(4, 6, 10, 11)]
cor(cor_mat)

# Classifying the dataset and splitting it into the reviews and stars
x <- reviews$text
y <- reviews$stars

# Data Cleaning
reviews$text <- as.String(reviews$text)


