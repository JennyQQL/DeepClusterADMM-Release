function out = netrecons(feature,w5,w6,w7,w8)
%this func can get reconstructed data with input feature.
N =size(feature,1);
w4probs = [feature  ones(N,1)];
w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(N,1)];
w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(N,1)];
w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(N,1)];
out = 1./(1 + exp(-w7probs*w8));

end