# DeepGradientCompression
It is implementation of Research paper "DEEP GRADIENT COMPRESSION: REDUCING THE COMMUNICATION BANDWIDTH FOR DISTRIBUTED TRAINING". Deep gradient compression is a technique by which the gradients are compressed before they are being sent. This approach greatly reduces the communication bandwidth and thus improves multi node training.

Motivation behind Deep Gradient Compression

To decrease the training time, we can increase the number of GPUs and computation. That means if we add 64 nodes, that means the training should be 64x times faster. In reality it is harder to achieve 64x speedup. The more the number of nodes, the more communication. 
Deep gradient compression is a technique by which the gradients are compressed before they are being sent. This approach greatly reduces the communication bandwidth and thus improves multi node training. 

Implementation
 
I implemented gradient sparsification approach. I sparsify the gradients by removing R% of gradients smaller than absolute threshold value, calling it as gradient dropping. This approach is slightly different from the approach proposed by Dryden et al. (2016) as I use a single absolute value of threshold, instead of dropping fixed number of positive and negative gradients separately. This approach is simpler and works just as well. There could be a case when the small gradients accumulate over the period of time and simple dropping them could damage convergence. Gradient accumulation is conducted to avoid losing important gradients. Detailed explanation on each task and corresponding experiments are mentioned below. All the experiments use GradientCompressionOptimizer, create by me which wraps arounds horovod’s distributed optimizer.  


Masking 
To test how the model behaves when gradients are dropped, masking was used. In masking all the absolute values less than the absolute threshold where set to zero and passed as it to the original optimizer. For example if the threshold in the below example is 1, all the indexes whose absolute value is less than 1 are set to 0.


Single Node - MNIST 
		
  Send k% grads | Accuracy  |	Loss   |
  --------------|-----------|--------|
  100 (send all)|	0.9908	  | 0.0285 |
  10	          | 0.9887	  | 0.0331 |
  5	            | 0.9873	  | 0.0381 |
  3	            | 0.9868	  | 0.0396 |
  2	            | 0.9852	  | 0.0446 |
  1	            | 0.9828	  | 0.0543 |
  0.9           |	0.9835	  | 0.0519 |
  0.5	          | 0.9832	  | 0.0468 |
  0.3	          | 0.9752	  | 0.08   |
  0.2	          | 0.9452	  | 0.1918 |
  0.1	          | 0.4099	  | 2.2113 |
  0 (Send None) |	0.2211	  | 2.247  |
		
		
		
Multi Node - MNIST 2 nodes		
		
  Send k% grads |	Accuracy |	Loss   |
  --------------|----------|---------|
  100(send all) |	0.9908	 | 0.021   | 
  95	          | 0.9905	 | 0.0266  |
  80	          | 0.9905	 | 0.0292  |
  70	          | 0.9898	 | 0.0306  |
  50	          | 0.9893	 | 0.0304  |
  30	          | 0.9882	 | 0.0345  |
  1	            | 0.9753	 | 0.0788  |
  0.5	          | 0.9765	 | 0.0772  |
  0.2	          | 0.9452	 | 0.1918  |
  0.1	          | 0.2385	 | 2.2694  |
  0 (Send None) |	0.1386	 | 2.2822  |
  
  
 The Library –
 
DeepGradientCompression optimizer wraps around tf.train.optimizer, which is the base class for optimizers. It overrides methods compute_gradients and apply_gradients. 

**compute_gradients** – gradients are computed using this method. It returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable". Note that "gradient" can be a Tensor, an IndexedSlices, or None if there is no gradient for the given variable. In this case the gradient would be an IndexedSlices object.

**apply_gradients** - Apply gradients to variables. This is the second part of minimize(). It returns an Operation that applies gradients.

How it can be used-

 

1. The optimizer file can be downloaded and added to the source code directory. 
2. Import the class in the model file. The model should be using Horovod distributed optimizer. 
3. Create an optimizer object and wrap it around it and call compute_gradients method. 
4. Call Sparse to dense method to convert the tensors to dense
5. Call apply gradients

 
 
*Use Cases –*
 
The optimizer could be used with any model using an optimizer. For example, I used the optimizer with BERT and MNIST. I am working integrating it with Resnet.
 
*Further steps-*
 

1. Optimizing Thresholding - I am going through a couple of research papers that suggest optimal way of finding threshold. I am trying to implement e sampling to reduce top-k selection time. Where I plan to sample only 0.1% to 1% of the gradients and perform top-k selection on the samples to estimate the threshold for the entire population. If the number of gradients exceeding the threshold is far more than expected, a precise threshold is calculated from the already-selected gradients. Hierarchically calculating the threshold significantly reduces top-k selection time. This would reduce the overall training time and still help maintain the accuracy.
2. Conduct experiments on BERT pre training 32 nodes.



