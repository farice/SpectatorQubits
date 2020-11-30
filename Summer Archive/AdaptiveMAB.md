Algorithm:

1. Start with an initial ‘gamma’ (hyper-paramter to tune) and a large action space for both contexts. 

2. Thompson sampling chooses a few arms after some iterations of this algorithm (calibration phase). 
The action space for the two contexts might be different now, so might have to treat these two cases separately. I suggest 
we change the code to have two indepdent action spaces (for two contexts) from the begining. 

3. After the calibration phase, we change the action space to be around the arms chosen by Thomson’s sampling.
We keep the number of arms fixed. 

4. At every step of the algorithm (after the calibration phase), calculate the density of errors (the 2nd spectator qubit
measurements should always be 0 when the correction fully compensates for the error, 1 otherwise.
We basically should look at the average time steps between two 1’s of the 2nd spectator qubit measurement.)
If this density (in some moving window) is higher than some threshold (we need to figure out what the threshold should be, 
another hyper-parameter to tune), we change our gamma (need to figure out if the update should be multiplicative or additive)
and we make our action space a little bigger (by how much should be function of this density). 
	