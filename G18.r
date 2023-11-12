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


forward <- function(nn, inp){
  ## Extract list h, W and b from nn
  h <- nn$h
  W <- nn$W
  b <- nn$b
  ## Get the number of layers.
  num_layers <- length(h)
  ## Set the first layer nodes equal to input values.
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

  ## Build a function to calculate the derivatives of L w.r.t hl(the node values for layer l)
  calculate_dh <- function(hl, k){
    ## set up a storage for derivatives
    dhl <- rep(0, length(hl))
    
    ## Loop over each node for layer l
    for (j in 1:length(hl)){
      ## Calculate the derivative
      if (j != k){
        dhl[j] <- exp(hl[j])/sum(exp(hl))
      } else{
        dhl[j] <- exp(hl[j])/sum(exp(hl)) - 1
      }
    }
    ## Return a vector of derivatives for layer l
    return(dhl)    
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
  
  ## Loop over layer
  for (l in 1:(num_layers-1)){
    hl <- as.vector(h[[l]])
    hl_1 <- as.vector(h[[l+1]])
    d <- calculate_dh(hl_1, k)
    d[hl_1 < 0] <- 0    
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

