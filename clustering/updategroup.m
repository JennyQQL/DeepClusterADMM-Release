function [batchBasis,C] = updategroup(clusterdata,CL,clustertargets,w1,w2,w3,w4)
%update the cluster assignment as well as renew the basises for cluster restrains.
N =size(clusterdata,1);
K = size(clustertargets,2);
C_dims = size(w4,2);
batchBasis = zeros(N,C_dims);
targetout = netcomput(clusterdata,w1,w2,w3,w4);%get hidden layer value
C = zeros(K,C_dims);
counter = zeros(1,K);
%directly get K cluster centre 'C' in hiden layer(or code layer) 
for i = 1:N
    C(CL(i),:) = C(CL(i),:)+ targetout(i,:);
    counter(CL(i)) = counter(CL(i))+1;
end
for i = 1:K
    C(i,:)= C(i,:)./ counter(i);
end

batchBasis(:, :) = C(CL, :);%copy centro by CL series for finetuning
end


