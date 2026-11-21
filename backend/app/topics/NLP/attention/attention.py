import numpy as np

## input vectors 
X = np.array([
    [1, 0],  # The
    [0, 1],  # cat
    [1, 1]   # drinks
])

## simplify the Q,K,V are the same 


Q = X
K = X
V = X


# step 1

K_T = K.T

# step 2 calculate the attention scores

scores = np.dot(Q, K_T)

## step 3 scale scores
dk = K.shape[1]
scaled_scores = scores / np.sqrt(dk)

##step 4 softmax

def softmax(x) :
    exp_x = np.exp(x - np.max(x , axis =-1 , keepdims=True))
    return exp_x / np.sum(exp_x , axis = -1 , keepdims= True)

attention_weights = softmax(scaled_scores)

## calculate scores

context = np.dot(attention_weights, V)

print("Q:\n", Q)
print("K:\n", K)
print("V:\n", V)

print("Attention Scores:\n", scores)
print("Scaled Scores:\n", scaled_scores)
print("Attention Weights:\n", attention_weights)
print("Context:\n", context)