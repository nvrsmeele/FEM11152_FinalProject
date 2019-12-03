#----
# Setup workspace
#----

# Load libraries
library(readr)
library(dplyr)
library(plyr)
library(caTools)

library(ggplot2)
library(tm)
library(SnowballC)

# Load full dataset
yelp_reviews <- read_csv("yelp_academic_dataset_review.csv")
yelp_business <- read_csv("yelp_academic_dataset_business.csv")

#----
# Data preprocessing
#----

##----
## Dataset "yelp_business"

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
## Create new dataframe "reviews" consisting of subset of all restaurants

# Merge dfs "business" and "yelp_reviews"
reviews_full <- inner_join(yelp_reviews, business)

# Create df "reviews" as subset of all restaurants
reviews_subset <- reviews_full[which(grepl("Restaurants", reviews_full$categories)),]

##----
## Dataset "reviews_subset"

# Check data structure
str(reviews_subset)

# (re-)Factor dependent variable "Stars"
reviews_subset$stars <- factor(reviews_subset$stars, ordered = TRUE)
reviews_subset$stars <- mapvalues(reviews_subset$stars, from = c("1", "2", "4", "5"),
                           to = c("1-2", "1-2", "4-5", "4-5"))

##----
## Create final dataset and split randomly for training/test

# Create dataset with n = 350,000 by random selection
set.seed(100)
samp <- sample(nrow(reviews_subset), size = 350000)
reviews <- reviews_subset[samp,]

# Create Train/Test set
set.seed(111)
samp <- sample.split(reviews$stars, SplitRatio = 2/3)
train <- subset(reviews, samp == TRUE)
test <- subset(reviews, samp == FALSE)

#----
# Text cleaning????








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


