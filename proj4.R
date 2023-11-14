# A function to initialize network
netup <- function(d) {
  ## Calculate the number of layers for the network.
  num_layers <- length(d)
  ## Set up a list for node values h.
  h <- lapply(d, function(num_nodes) rep(0, num_nodes))
  
  ## define a function to initialize weight parameter matrix W linking layer l to layer l+1.
  initialize_W <- function(d, l){
    ## Generate W values with U(0, 0.2) random deviates.
    w_values <- runif(d[l]*d[l+1], 0, 0.2)
    ## Build up the matrix W.
    w_matrix <- matrix(w_values, d[l+1], d[l])
    return(w_matrix)
  }
  ## Using 'initialize_W' function to initialize all matrix W.
  W <- lapply(1:(num_layers-1), function(l) initialize_W(d, l))
  
  ## Initialize all offset parameters b linking layer l to l+1.
  b <- lapply(1:(num_layers-1), function(l) runif(d[l+1], 0, 0.2))
  
  ## Return a list containing node values(h), weight matrices(W) and offset vectors(b).
  return(list(h = h, W = W, b = b))
}


# A function to update nodes for every new data
forward <- function(nn, inp){
  #inp: a vector of input values for the first layer
  ## Extract list h, W and b from nn
  h <- nn$h
  W <- nn$W
  b <- nn$b
  ## Get the number of layers.
  num_layers <- length(h)
  ## Set the node values h for the first layer equal to input values.
  h[[1]] <- unlist(inp)
  
  ## Loop over remaining layers.
  for (l in 2:num_layers) {
    # Compute the node values for the current layer.
    h[[l]] <- pmax(0, as.matrix(W[[l-1]]) %*% as.vector(h[[l-1]]) + as.matrix(b[[l-1]]))
  }
  ## Return the updated network list.
  return(list(h = h, W = W, b = b))
}



backward <- function(nn, k) {
  #k is a scalar(类别)

  ## Build a function to calculate the derivatives of L w.r.t hL(the node values for layer l)
  calculate_dh <- function(hL, k){
    ## set up a storage for derivatives
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
  ## Calculate the number of layers for the network.
  num_layers <- length(h)  
  ## Set up lists for derivatives w.r.t. the nodes, weights and offsets
  dh <- list()
  dW <- list()
  db <- list()
  
  ## calculate the derivative of last layer
  dh[[num_layers]] <- calculate_dh(h[[num_layers]], k)
  ## Loop over layer
  for (l in (num_layers-1):1){
  ## calculate the derivative of previous layer
    hl <- as.vector(h[[l]])
    hl_1 <- as.vector(h[[l+1]])
    d <- dh[[l+1]]
    d[hl_1 <= 0] <- 0
    db[[l]] <- d
    dh[[l]] <- t(as.matrix(W[[l]])) %*% d
    dW[[l]] <- d %*% t(hl)
  }
  
  # Update the network list
  nn$dh <- dh 
  nn$dW <- dW
  nn$db <- db
  return(nn)
} 

train <- function(nn,inp,k,eta=0.01,mb=10,nstep=10000){
  num_data <- nrow(inp)
  num_layers <- length(nn$h)
  for (step in 1:nstep){
    ##Sample a minibatch of training data
    sample_index <- sample(1:num_data, mb, replace = TRUE)
    sample_data <- inp[sample_index, , drop=FALSE]
    sample_class <- k[sample_index]
    
    # Set up storages for the sum of gradients for one step
    dW_sum <- rep(list(0), num_layers-1)
    db_sum <- rep(list(0), num_layers-1)
    
    for (i in 1:mb){
      ##Go forward 
      network <- forward(nn, sample_data[i,])
      ##Go backward
      network <- backward(network, sample_class[i])
      ##Update parameters
      dW_sum <- Map('+', dW_sum, network$dW)
      db_sum <- Map('+', db_sum, network$db)
    }
    ##Average gradient
    dW_change <- Map('*', dW_sum, eta/mb)
    db_change <- Map('*', db_sum, eta/mb)
    nn$W <- Map('-', nn$W, dW_change)
    nn$b <- Map('-', nn$b, db_change)
  }
  return(nn)
}

# Give species numeric class
iris$class <- as.numeric((iris$Species))
# Split data into testing data and training data
index <- seq(5,nrow(iris),5)
test <- iris[index,1:4]
training <- iris[-index,1:4]
k <- iris[-index,6]
inp=training
# Set the seed to provide an example in which training has worked and the loss has been substantially reduced from pre- to post-training
set.seed(n)
d=c(4,8,7,3)
nn=netup(d)
final_network=train(nn,inp,k,eta=0.01,mb=10,nstep=10000)

# Define a function to classify the test data
predict_species <- function(network, test_data) {
  predictions <- sapply(1:nrow(test_data), function(i) {
    output <- forward(network, test_data[i,])
    predicted_class <- which.max(output$h[[length(output$h)]])
    return(predicted_class)
  })
  return(predictions)
}

# Classify the test data using the trained network
predicted_classes <- predict_species(final_network, test)

# The actual label
actual_classes <- iris[index, 6]

# Compute the misclassification rate
misclassification_rate <- sum(predicted_classes != actual_classes) / length(actual_classes)

print(misclassification_rate)
