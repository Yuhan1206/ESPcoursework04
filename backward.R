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