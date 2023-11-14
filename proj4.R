# Group member: Yuhan Liu unn: s2589828, Haojia Li unn: s2604554, Xinyi Wang unn:s2529097
# Address of github repo:
# Team member distribution: All team members participated in discussing plans and debugging code for the whole neural network.
# Haojia Li: 
# Yuhan Liu: 
# Xinyi Wang: 

# A function to initialize network

netup <- function(d) {
  ## Calculate the number of layers for the network
  num_layers <- length(d)
  ## Initialize the node value 'h' for each layer. For each element in 'd', create a zero vector of the corresponding length
  ## 'h' will be a list where each element is a vector representing the node values of the corresponding layer
  h <- lapply(d, function(num_nodes) rep(0, num_nodes))
  
  ## define a function to initialize weight parameter matrix W linking layer l to layer l+1
  initialize_W <- function(d, l){
    ## Generate W values with U(0, 0.2) random deviates. The length of the vector is the total number of elements in the weight matrix
    w_values <- runif(d[l]*d[l+1], 0, 0.2)
    ## Convert vectors into a matrix where the number of rows is the number of nodes in the next layer 
    ## and the number of columns is the number of nodes in the current layer
    w_matrix <- matrix(w_values, d[l+1], d[l])
    return(w_matrix)
  }
  ## Using 'initialize_W' function to initialize all matrix W.
  W <- lapply(1:(num_layers-1), function(l) initialize_W(d, l))
  
  ## Initialize all offset parameters b linking layer l to l+1 with U(0, 0.2) random deviates
  b <- lapply(1:(num_layers-1), function(l) runif(d[l+1], 0, 0.2))
  
  ## Return a list containing node values(h), weight matrices(W) and offset vectors(b)
  return(list(h = h, W = W, b = b))
}


# A function to update nodes for every new data
# inp: a vector of input values for the first layer

forward <- function(nn, inp){
  ## Extract list h, W and b from nn
  h <- nn$h
  W <- nn$W
  b <- nn$b
  ## Get the number of layers
  num_layers <- length(h)
  ## Set the node values h for the first layer equal to input values
  h[[1]] <- unlist(inp)
  
  ## Loop over remaining layers
  for (l in 2:num_layers) {
    ## Use ReLU transform and compute the node values for the current layer
    h[[l]] <- pmax(0, as.matrix(W[[l-1]]) %*% as.vector(h[[l-1]]) + as.matrix(b[[l-1]]))
  }
  ## Return the updated network list
  return(list(h = h, W = W, b = b))
}



backward <- function(nn, k) {
  ## k is a scalar(类别)

  ## Build a function to calculate the derivatives of L w.r.t hL(the node values for layer l)
  calculate_dh <- function(hL, k){
    ## Initialize a zero vector of the same length as the number of nodes in the last layer to store the derivatives
    dhL <- rep(0, length(hL))
    
    ## Loop over each node for layer L (the last layer)
    for (j in 1:length(hL)){
      ## Calculate the derivative
      if (j != k){
        dhL[j] <- exp(hL[j])/sum(exp(hL))
      } else{
        dhL[j] <- exp(hL[j])/sum(exp(hL)) - 1
      }
    }
    ## Return a vector of derivatives for layer L
    return(dhL)    
  }
  
  ## Extract list h and W from nn
  h <- nn$h  
  W <- nn$W
  ## Calculate the number of layers for the network
  num_layers <- length(h)  
  ## Set up lists for derivatives w.r.t. the nodes, weights and offsets
  dh <- list()
  dW <- list()
  db <- list()
  
  ## Calculate the derivative of last layer
  dh[[num_layers]] <- calculate_dh(h[[num_layers]], k)
  ## Loop backwards to calculate the derivative of each layer
  for (l in (num_layers-1):1){
    ## Extract the node values of the current layer and the next layer and the derivative of the next layer
    hl <- as.vector(h[[l]])
    hl_1 <- as.vector(h[[l+1]])
    d <- dh[[l+1]]
    ## Use the ReLU transform
    d[hl_1 <= 0] <- 0
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


train <- function(nn,inp,k,eta=0.01,mb=10,nstep=10000){
  ## Calculate the number of rows and the number of network layers
  num_data <- nrow(inp)
  num_layers <- length(nn$h)
  ## Loop through each training step
  for (step in 1:nstep){
    ## Sample a minibatch of training data
    sample_index <- sample(1:num_data, mb)
    ## Extract a minibatch of samples from the input data matrix and category label vectors according to the sample index
    sample_data <- inp[sample_index, , drop=FALSE]
    sample_class <- k[sample_index]
    
    # Set up storages for the sum of gradients for one step
    dW_sum <- rep(list(0), num_layers-1)
    db_sum <- rep(list(0), num_layers-1)

    ## Loop over each data point in the small batch
    for (i in 1:mb){
      ## Go forward 
      network <- forward(nn, sample_data[i,])
      ## Go backward
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


## Convert Species from iris dataset to numerical form
iris$class <- as.numeric((iris$Species))
## Split data into testing data and training data
## The test data consists of every 5th row of the iris dataset, starting from row 5
index <- seq(5,nrow(iris),5)
## Extract test data from iris dataset. Only the first four columns (the four characteristics of iris flowers) are taken
test <- iris[index,1:4]
## Extract the training data, that is, all the data except the test data. Again, only take the first four columns 
training <- iris[-index,1:4]
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


## Define a function to classify the test data
predict_species <- function(network, test_data) {
  ## Use the sapply function to operate on each row of the test data
  predictions <- sapply(1:nrow(test_data), function(i) {
    ## Use forward function to obtain the network output of the current test sample
    output <- forward(network, test_data[i,])
    ## Obtain the node values of the output layer and convert them to vector
    last_layer <- as.vector(output$h[[length(output$h)]])
    ## Calculate the probability for each category
    p <- exp(last_layer)/sum(exp(last_layer))
    ## Find the class with the highest probability
    predicted_class <- which.max(p)
    return(predicted_class)
  })
  return(predictions)
}

## Classify the test data using the trained network
predicted_classes <- predict_species(final_network, test)

## Obtain the actual labels
actual_classes <- iris[index, 6]

## Compute the misclassification rate
misclassification_rate <- sum(predicted_classes != actual_classes) / length(actual_classes)

print(misclassification_rate)
