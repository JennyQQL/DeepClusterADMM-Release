function [batchBasis, centro, RL] = randinitial_gmm(clusterdata,clustertargets,w1,w2,w3,w4)
%Get rand label and centers for clustering as well as the real label for evaluating.
rand('state',0);
N = size(clusterdata,1);
K = size(clustertargets,2);
D_dims = size(clusterdata,2);
C_dims = size(w4,2);
batchBasis = zeros(N,C_dims);
out = netcomput(clusterdata,w1,w2,w3,w4);
[CLK,centro] =kmeans(out,K);
[~, RL] = max(clustertargets, [], 2);
batchBasis(:, :) = centro(CLK, :);%copy centre by CLK series for rand initial training.
end


