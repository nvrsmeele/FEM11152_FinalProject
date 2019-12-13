#----
# Setup workspace
#----

# Load libraries
library(readr)
library(dplyr)
library(plyr)
library(caTools)
library(tm)
library(textstem)
library(tokenizers)
library(lexicon)
library(tidytext)
library(naivebayes)
library(caret)
library(randomForest)
library(rpart)
library(xgboost)
library(readr)
library(stringr)
library(car)

#----------------------
library(tokenizers)
library(SentimentAnalysis)
library(ggplot2)
library(SnowballC)
#----------------------

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
business <- business[, c(-5:-27, -29:-31, -33:-71)]

# Rename columns
business$business_name <- business$name
business <- business[,-3]

business$avg_stars <- business$stars
business <- business[,-5]

# Split character value in "Categories" variable
business$categories <- strsplit(business$categories, split = ";")

##----
## 1.2 Create new dataframe "reviews" consisting of subset of all restaurants

# Merge dfs "business" and "yelp_reviews"
reviews_full <- inner_join(yelp_reviews, business)

# Create df "reviews" as subset of all restaurants
reviews_subset <- reviews_full[which(grepl("Diners", reviews_full$categories)),]

##----
## 1.3 Dataset "reviews_subset" and create workable dataset as "reviews_final"

# Check data structure
str(reviews_subset)

# (re-)Factor dependent variable "Stars"
reviews_subset$stars <- factor(reviews_subset$stars, ordered = TRUE)

# Create df with dependent variable "Stars" as 3 classes
#reviews_3class <- reviews_subset
#reviews_subset$stars <- mapvalues(reviews_subset$stars, from = c("1", "2", "4", "5"),
#                           to = c("1-2", "1-2", "4-5", "4-5"))

# Create dataset "reviews_final" with n = 350,000 by random selection
#set.seed(100)
#samp <- sample(nrow(reviews_subset), size = 75000)
#reviews_final <- reviews_subset[samp,]


# Clean work environment to free memory
rm(business, reviews_full, yelp_business, yelp_reviews, reviews_subset,samp)

##----
## 1.4 Text data preprocessing (dataset "reviews_subset")

# Create a VCorpus
#text <- VCorpus(VectorSource(reviews_final$text))
text <- VCorpus(VectorSource(reviews_subset$text))

# Corpus text cleaning
text <- tm_map(text, stripWhitespace)
text <- tm_map(text, content_transformer(tolower))
text <- tm_map(text, removeWords, stopwords("english"))
text <- tm_map(text, removePunctuation)
text <- tm_map(text, removeNumbers)
text_stem <- tm_map(text, stemDocument)

#----
# Concept for Lemmatization
#----
text_lemma <- tm_map(text, lemmatize_words)
text_vec <- tidy(text)
text_vec$text <- as.String(text_vec$text)
text_vec$id <- as.integer(text_vec$id)

text_lemma <- text_vec %>%
  lemmatize_strings(text, dictionary = lexicon::hash_nrc_emotions)

bistem_textDTM <- text_tidy %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  group_by(id) %>%
  dplyr::count(bigram) %>%
  bind_tf_idf(bigram, id, n) %>%
  cast_dtm(id, bigram, tf_idf)
#----

# Create Corpus + text cleaning for 3 classes
#----
text3 <- VCorpus(VectorSource(reviews_final3$text))
text3 <- tm_map(text3, stripWhitespace)
text3 <- tm_map(text3, content_transformer(tolower))
text3 <- tm_map(text3, removeWords, stopwords("english"))
text3 <- tm_map(text3, removePunctuation)
text3 <- tm_map(text3, removeNumbers)
text_stem3 <- tm_map(text3, stemDocument)
#----

##----
## 1.5 Ngram modeling based on tf-idf, and remove sparse terms

# Unigram model based on Stemming (base model)
unistem_textDTM <- DocumentTermMatrix(text_stem,
                   control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
sparse <- removeSparseTerms(unistem_textDTM, 0.98) # remove sparse terms
unistem_textDTM <- as.data.frame(as.matrix(sparse)) # convert dtm to matrix
unistem_textDTM$stars <- reviews_subset$stars # add dependent variable to matrix

# Bigram model based on Stemming
library(RWeka)
bigramToken <- function(x){
  tokenize_ngrams(x, n = 2)
}


bistem_textDTM <- DocumentTermMatrix(text_stem,
                  control = list(tokenize = bigramToken,
                  weighting = function(x) weightTfIdf(x, normalize = FALSE)))
sparse <- removeSparseTerms(unistem_textDTM, 0.99) # remove sparse terms
unistem_textDTM <- as.data.frame(as.matrix(sparse)) # convert dtm to matrix
unistem_textDTM$stars <- reviews_final$stars # add dependent variable to matrix

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
bistem_textDTM$stars <- reviews_final$stars # add dependent variable to matrix

# Unigram model based on Lemmatization

# Bigram model based on Lemmatization


#----
# 2. Model building preparation
#----

##----
## Split unistem_textDTM randomly for training/test
set.seed(111)
samp <- sample.split(unistem_textDTM$stars, SplitRatio = 2/3)
train.uni <- subset(unistem_textDTM, samp == TRUE) # declare training set
test.uni <- subset(unistem_textDTM, samp == FALSE) # declare test set

ytrain.uni <- train.uni[,500] # declare training response variable
xtrain.uni <- train.uni[,1:499] # declare training predictor variables

trainDown <- downSample(xtrain.uni, ytrain.uni, yname = "stars")
xtrainDown.uni <- trainDown[,1:499]
ytrainDown.uni <- trainDown[,500]

ytest.uni <- test.uni[,888] # declare test response variable
xtest.uni <- test.uni[,1:887] # declare test predictor variables

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
## Clean work environment to free memory
rm(text, text_stem, sparse, text_tidy, samp)

##----
## Separate dependent/independent variables in training set???
x_train <- train[,1:887]
y_train <- train[,888]

train_down <- downSample(x_train, y_train, yname = "Stars")

x_trainDown <- train_down[,1:887]
y_trainDown <- train_down[,888]


#----
# 3. Model building
#----

##----
## multi-class Naive Bayes Classifier (base model)

# Step1: Create NB classifier
nb_model <- multinomial_naive_bayes(as.matrix(xtrain.uni), ytrain.uni)

# Step 2: Generate predictions of the NB classifier
pred <- predict(nb_model, data = test.uni, type = "class")

# Step 3: Create confusion matrix of NB's prediction performance
nb_cfm <- confusionMatrix(data = pred, reference = ytrain.uni)

##----
## Random Forest Classifier

# Step 1: Run initial Random Forest model
set.seed(200)
rf_model <- randomForest(xtrainDown.uni, ytrainDown.uni, mtry = 22, replace = TRUE,
                         importance = TRUE, ntree = 75, do.trace = TRUE,
                         control = rpart.control(minsplit = 2, cp = 0))

pred <- predict(rf_model, data = test.uni, type = "class")
rf_cfm <- confusionMatrix(data = pred, reference = ytrainDown.uni)

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

xgbtrain <- xgb.DMatrix(data = as.matrix(xtrainDown.uni), label = as.numeric(ytrainDown.uni)-1)

numberOfClasses <- length(unique(ytrainDown.uni))
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = numberOfClasses)
nround    <- 50 # number of XGBoost rounds
cv.nfold  <- 5

# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = xgbtrain, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = ytrainDown.uni)

confusionMatrix(factor(OOF_prediction$max_prob),
                factor(OOF_prediction$label),
                mode = "everything")


##----
## Neural Network Classifier

library(keras)
library(tensorflow)
library(reticulate)

reticulate::use_python("/opt/anaconda3/envs/r-base/bin")

xtrain.norm <- normalize(as.matrix(xtrain.uni))
ytrain.bin <- to_categorical(as.numeric(ytrain.uni)-1, 5)
ytest.bin <- to_categorical(as.numeric(ytest.uni)-1, 5)

model <- keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "relu", input_shape = c(499)) %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(0.25) %>%
  layer_dense(units = 8, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = list('accuracy')
)

trained_model <- model %>% fit(
  xtrain.norm,
  ytrain.bin,
  epochs = 50,
  batch_size = 512,
  validation_split = 0.2,
)






#----
# https://www.kaggle.com/amhchiu/bag-of-ingredients-in-r
#----

# Create Training/Test
temp <- sample(nrow(reviews), size=233333)
train <- reviews[temp,]
test <- reviews[-temp,]

# Create Corpus
text <- Corpus(VectorSource(train$text))

# Cleaning
text <- tm_map(text, stripWhitespace)
text <- tm_map(text, content_transformer(tolower))
text <- tm_map(text, removeWords, stopwords("english"))
text <- tm_map(text, stemDocument)

# Create Document Term Matrix
textDTM <- DocumentTermMatrix(text)

# Feature selection
sparse <- removeSparseTerms(textDTM, 0.99)
## This function takes a second parameters, the sparsity threshold.
## The sparsity threshold works as follows.
## If we say 0.98, this means to only keep terms that appear in 2% or more of the recipes.
## If we say 0.99, that means to only keep terms that appear in 1% or more of the recipes.

textDTM <- as.data.frame(as.matrix(sparse))
## Add the dependent variable to the data.frame
textDTM$stars <- as.factor(train$stars)

# Create Model
x <- sample(nrow(textDTM), size = 77778)
intrain <- textDTM[x,]
intest <- textDTM[-x,]

library(naivebayes)

x <- as.matrix(intrain[,1:891])
y <- intrain[,892]

model <- multinomial_naive_bayes(x, y)
pred <- predict(model, data = intest, type = "class")

library(caret)

cfm <- confusionMatrix(pred, intrain$stars)
cfm


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


