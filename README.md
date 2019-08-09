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
