#----
# Setup workspace
#----

# Load libraries
library(readr)
library(ggplot2)
library(dplyr)
library(tm)
library(SnowballC)

# Load full dataset
yelp_reviews <- read_csv("yelp_academic_dataset_review.csv")

# Create random subset dataset
set.seed(100)
samp <- sample(nrow(yelp_reviews), size = 350000)
reviews <- yelp_reviews[samp, ]

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


