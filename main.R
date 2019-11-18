#----
# Setup workspace
#----

# Load libraries
library(readr)
library(ggplot2)
library(dplyr)

# Load full dataset
yelp_reviews <- read_csv("Data/yelp_academic_dataset_review.csv")

# Create random subset dataset
set.seed(100)
samp <- sample(nrow(yelp_reviews), size = 350000)
reviews <- yelp_reviews[samp, ]

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


