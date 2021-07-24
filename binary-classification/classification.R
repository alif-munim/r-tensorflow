# Install Packages
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("EBImage")

# Load Packages
library(EBImage)
library(keras)

# Set workdir
setwd("C:\\Users\\alif-\\Desktop\\binary-classification\\images")
pics <- c("p1.jpg", "p2.jpg", "p3.jpg", "p4.jpg", "p5.jpg", "p6.jpg",
          "c1.jpg", "c2.jpg", "c3.jpg", "c4.jpg", "c5.jpg", "c6.jpg")
mypic <- list()
for (i in 1:12) {mypic[[i]] <- readImage(pics[i])}

# Explore data
print(mypic[[1]])
display(mypic[[8]])
summary(mypic[[1]])
hist(mypic[[2]])
str(mypic)

# Resize images
for (i in 1:12) {mypic[[i]] <- resize(mypic[[i]], 28, 28)}

# Unroll image matrices into a single vector
for (i in 1:12) {mypic[[i]] <- array_reshape(mypic[[i]], c(28, 28, 3))}

# Combine data using row bind
train_x <- NULL

# Add first 5 plane images
for (i in 1:5) {train_x <- rbind(train_x, mypic[[i]])}
str(train_x)

# Add first 5 car images
for (i in 7:11) {train_x <- rbind(train_x, mypic[[i]])}
str(train_x)

# Create test examples
test_x <- rbind(mypic[[6]], mypic[[12]])

# Set ground truth (dependent variables y)
train_y <- c(0,0,0,0,0,1,1,1,1,1)

# Set test ground truth
test_y <- c(0, 1)

# One hot encoding
train_labels <- to_categorical(train_y)
test_labels <- to_categorical(test_y)

# Create model
model <- keras_model_sequential()
model %>%
  layer_dense(units=256, activation = "relu", input_shape = c(2352)) %>%
  layer_dense(units=128, activation = "relu") %>%
  layer_dense(units=2, activation = "softmax")
summary(model)

# Compile model
model %>%
  compile(loss = "binary_crossentropy",
          optimize = optimizer_rmsprop(),
          metrics = c("accuracy"))

# Fit model
history <- model %>%
  fit(train_x, train_labels, epochs = 30, batch_size = 32,
      validation_split = 0.2)

# Plot history
plot(history)

# Evaluation and predicted labels on train data
model %>% evaluate(train_x, train_labels)
pred <- model %>% predict_classes(train_x)

# Confusion matrix for train data
table(Predicted = pred, Actual = train_y)

# Probability values for train data
prob <- model %>% predict_proba(train_x)
cbind(prob, Predicted = pred, Actual = train_y)

# Evaluation and prediction on test data
model %>% evaluate(test_x, test_labels)
pred <- model %>% predict_classes(test_x)

# Confusion matrix for predictions and test labels
table(Predicted = pred, Actual = test_y)
