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
library(lexicon)
library(tidytext)
library(naivebayes)
library(caret)


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
reviews_subset <- reviews_full[which(grepl("Restaurants", reviews_full$categories)),]

##----
## 1.3 Dataset "reviews_subset" and create workable dataset as "reviews_final"

# Check data structure
str(reviews_subset)

# (re-)Factor dependent variable "Stars"
reviews_subset$stars <- factor(reviews_subset$stars, ordered = TRUE)

# Create df with dependent variable "Stars" as 3 classes
reviews_3class <- reviews_subset
reviews_3class$stars <- mapvalues(reviews_subset$stars, from = c("1", "2", "4", "5"),
                           to = c("1-2", "1-2", "4-5", "4-5"))

# Create dataset "reviews_final" with n = 350,000 by random selection
set.seed(100)
samp <- sample(nrow(reviews_subset), size = 350000)
reviews_final <- reviews_subset[samp,]
reviews_final3 <- reviews_3class[samp,]

# Clean work environment to free memory
rm(business, reviews_full, yelp_business, yelp_reviews, reviews_subset, reviews_3class,samp)

##----
## 1.4 Text data preprocessing (dataset "reviews_subset")

# Create a VCorpus
text <- VCorpus(VectorSource(reviews_final$text))

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
text3 <- VCorpus(VectorSource(reviews_final3$text))
text3 <- tm_map(text3, stripWhitespace)
text3 <- tm_map(text3, content_transformer(tolower))
text3 <- tm_map(text3, removeWords, stopwords("english"))
text3 <- tm_map(text3, removePunctuation)
text3 <- tm_map(text3, removeNumbers)
text_stem3 <- tm_map(text3, stemDocument)

##----
## 1.5 Ngram modeling based on tf-idf, and remove sparse terms

# Unigram model based on Stemming (base model)
unistem_textDTM <- DocumentTermMatrix(text_stem,
                   control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))
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
## Split dataset randomly for training/test
set.seed(111)
samp <- sample.split(unistem_textDTM$stars, SplitRatio = 2/3)
train <- subset(unistem_textDTM, samp == TRUE)
test <- subset(unistem_textDTM, samp == FALSE)

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
## multi-class Naive Bayes Classifier
model <- multinomial_naive_bayes(as.matrix(x_trainDown), y_trainDown)
pred <- predict(model, data = test, type = "class")
cfm <- confusionMatrix(data = pred, reference = y_train)
cfm

##----
## Random Forest Classifier


##----
## XGBoost Classifier


##----
## Neural Network Classifier























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


