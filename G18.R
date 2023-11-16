# Group member: Yuhan Liu unn: s2589828, Haojia Li unn: s2604554, Xinyi Wang unn:s2529097
# Address of github repo: https://github.com/Yuhan1206/ESPcoursework04
# Team member distribution: All team members participated in discussing plans and debugging code for the whole neural network.
# Haojia Li: create netup, forward and backward functions and write comments (34%)
# Yuhan Liu: create backward, train functions and write comments (33%)
# Xinyi Wang: create predict function and write comments (33%)

# This code is used to create a neural network with given nodes number for each layer.
# First, we initialise the network h, W, and b by using uniform random numbers in function netup.
# Then, we update all nodes' values when given the input value in function forward.
# Next, we calculate the gradients of loss function in order to update W and b step by step in funciton backward.
# After all the pre-work, we train the network using function train
# and predict the class of test data using function predict_species.

# A function to initialize network
netup <- function(d) {
  # Inputs:
  #   d - a vector which contains the nodes number of each layer
  # Outputs:
  #   a list which contains the initial h,W,b of each layer
  
  ## Calculate the number of layers for the network
  num_layers <- length(d)
  ## Initialize the node value 'h' for each layer.
  ## 'h' will be a list where each element is a vector representing the node values of the corresponding layer
  h <- lapply(d, function(num_nodes) rep(0, num_nodes))
  
  ## A function to initialize weight parameter matrix W linking layer l to layer l+1
  initialize_W <- function(d, l){
    ## Generate W values with U(0, 0.2) random deviates.
    w_values <- runif(d[l]*d[l+1], 0, 0.2)
    ## Convert vectors into a matrix where rows number is the nodes number in the next layer 
    ## and columns number is the nodes number in the current layer
    w_matrix <- matrix(w_values, d[l+1], d[l])
    return(w_matrix)
  }
  ## Using 'initialize_W' function to initialize all matrix W.
  W <- lapply(1:(num_layers-1), function(l) initialize_W(d, l))
  
  ## Initialize all offset parameters b linking layer l to l+1
  b <- lapply(1:(num_layers-1), function(l) runif(d[l+1], 0, 0.2))
  
  ## Return a list containing node values(h), weight matrices(W) and offset vectors(b)
  return(list(h = h, W = W, b = b))
}


# A function to update nodes for every new data
forward <- function(nn, inp){
  # Inputs:
  #  nn - the network that returns from netup function
  #  inp - a vector of input values for the first layer
  # Outputs:
  #  a list which contains the h,W,b of each layer after giving inp
  
  ## Extract list h, W and b from nn
  h <- nn$h
  W <- nn$W
  b <- nn$b
  ## Get the number of layers
  num_layers <- length(h)
  ## Set the node values h for the first layer equal to input values
  h[[1]] <- inp
  
  ## Loop over remaining layers 
  for (l in 2:num_layers) {
  ## Use ReLU transform and compute the node values for the current layer
    h[[l]] <- pmax(0, W[[l-1]] %*% h[[l-1]] + b[[l-1]])
  }
  ## Return the updated network list
  return(list(h = h, W = W, b = b))
}


# A function to calculate the derivatives of the loss
backward <- function(nn, k) {
  # Inputs:
  #  nn - the network that returns from forward function
  #  k - a scalar class of input data
  # Outputs:
  #  a list which contains the h,W,b,dh,dW,db of each layer

  ## Extract list h and W from nn
  h <- nn$h  
  W <- nn$W
  ## Calculate the number of layers for the network
  num_layers <- length(h)  
  ## Set up lists for derivatives w.r.t. the nodes, weights and offsets
  dh <- vector("list", num_layers)
  dW <- vector("list", num_layers-1)
  db <- vector("list", num_layers-1)
  ## Calculate the derivative of h in the last layer
  dh[[num_layers]] <- exp(h[[num_layers]])/sum(exp(h[[num_layers]]))
  dh[[num_layers]][k] <- dh[[num_layers]][k] - 1

  ## Loop backwards to calculate the derivative of each layer
  for (l in (num_layers-1):1){
    ## Extract the node values of the current layer and the next layer
    hl <- h[[l]]
    hl_1 <- h[[l+1]]
    ## Calculate the d values using dh in the next layer
    d <- dh[[l+1]]
    d[hl_1 <= 0] <- 0
    ## Calculate db,dh,dW
    ## Every time, we use d values of the next layer to calculate db,dh,dW of current layer
    ## Then, we update d values using dh in the next loop and calculate db,dh,dW again
    db[[l]] <- d
    dh[[l]] <- t(as.matrix(W[[l]])) %*% d
    dW[[l]] <- d %*% t(hl)
  }
  
  ## Update the network list
  nn$dh <- dh 
  nn$dW <- dW
  nn$db <- db
  return(nn)
} 

# A function to train the whole network
train <- function(nn,inp,k,eta=0.01,mb=10,nstep=10000){
  # Inputs:
  #  nn - the network that returns from netup function
  #  k - a vector which contains the class of training data
  #  inp - a dataframe which contains the training data
  #  eta - the step size
  #  mb - the number of data to randomly sample to compute the gradient
  #  nstep - the number of optimization steps to take
  # Outputs:
  #  nn - the final network
  
  ## Calculate the number of rows and the number of network layers
  num_data <- nrow(inp)
  num_layers <- length(nn$h)
  ## Loop through each training step
  for (step in 1:nstep){
    ## For each step we just use a small sample of training datas to calculate the average gradients as this step's gradients
    ## Sample a minibatch of training data from the input data matrix and category label vectors
    sample_index <- sample(1:num_data, mb)
    sample_data <- inp[sample_index, , drop=FALSE]
    sample_class <- k[sample_index]
    
    # Set up storages for the sum of gradients for one step
    dW_sum <- rep(list(0), num_layers-1)
    db_sum <- rep(list(0), num_layers-1)

    ## Loop over each data point in the small batch
    for (i in 1:mb){
      ## Go forward to update all nodes values
      network <- forward(nn, sample_data[i,])
      ## Go backward to calculate the derivatives of parameters
      network <- backward(network, sample_class[i])
      ## Accumulate the gradient of the current data point to the sum of gradients
      dW_sum <- Map('+', dW_sum, network$dW)
      db_sum <- Map('+', db_sum, network$db)
    }
    ## Calculate the average gradient and multiply by the learning rate
    dW_change <- Map('*', dW_sum, eta/mb)
    db_change <- Map('*', db_sum, eta/mb)
    ## Use gradient descent to update parameters by subtracting the gradient change from the current value
    nn$W <- Map('-', nn$W, dW_change)
    nn$b <- Map('-', nn$b, db_change)
  }
  return(nn)
}

# Define a function to classify the test data
predict_species <- function(network, test_data) {
  # Input:
  #  network: a network return from train funciton
  #  test_data: a data frame contains all test data
  # Output:
  #  predicted_class: the predict class of test data
  
  ## Use the sapply function to operate on each row of the test data
  predictions <- sapply(1:nrow(test_data), function(i) {
    ## Use forward function to obtain the network output of the current test sample
    output <- forward(network, test_data[i,])
    ## Obtain the node values of the output layer and convert them to vector
    last_layer <- output$h[[length(output$h)]]
    ## Calculate the probability for each category
    p <- exp(last_layer)/sum(exp(last_layer))
    ## Find the class with the highest probability
    predicted_class <- which.max(p)
    return(predicted_class)
  })
  return(predictions)
}
              
## Convert Species from iris dataset to numerical form
iris$class <- as.numeric((iris$Species))
## Split data into testing data and training data
## The test data consists of every 5th row of the iris dataset, starting from row 5
index <- seq(5,nrow(iris),5)
test <- as.matrix(iris[index,1:4])
training <- as.matrix(iris[-index,1:4])
## Extract the category labels
k <- iris[-index,6]
              
## Define the structure of the neural network
inp=training
d=c(4,8,7,3)
## Set the seed
set.seed(1)
## Use the 'netup' function to initialize the neural network
nn=netup(d)
## Use the 'train' function to train the neural network
final_network=train(nn,inp,k,eta=0.01,mb=10,nstep=10000)
              
## Classify the test data using the trained network
predicted_classes <- predict_species(final_network, test)
## Obtain the actual labels
actual_classes <- iris[index, 6]
## Compute the misclassification rate
misclassification_rate <- sum(predicted_classes != actual_classes) / length(actual_classes)
print(misclassification_rate)
