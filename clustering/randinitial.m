function [batchBasis, centro, RL] = randinitial(clusterdata,clustertargets,w1,w2,w3,w4)
%Get rand label and centers for clustering as well as the real label for evaluating.
%rand('state',0);
N = size(clusterdata,1);
K = size(clustertargets,2);
D_dims = size(clusterdata,2);
C_dims = size(w4,2);
batchBasis = zeros(N,C_dims);
[~, RL] = max(clustertargets, [], 2);%get real label from clustertargets, only used for computing pur, acc & nmi.
CLK = randperm(N);            %compute rand label
CLK = mod(CLK,K)+1;
CLK=CLK';
% 
% %compute C across all dataset
% %get K cluster centre 'C' in visual layer(or bottom layer)
C = zeros(K,D_dims);
counter = zeros(1,K);
for i = 1:N
     C(CLK(i),:) = C(CLK(i),:)+clusterdata(i,:);
     counter(CLK(i)) = counter(CLK(i))+1;
end
for i = 1:K
  C(i,:)= C(i,:)./ counter(i);
end
%  
if size(clusterdata,2)==2000
     centro = netcomput_R(C,w1,w2,w3,w4);% for reuters dataset
else
     centro = netcomput(C,w1,w2,w3,w4);%get centre from 'C' through the encoder nets
end
%[CLK,centro] = kmeans(clusterdata,K);

batchBasis(:, :) = centro(CLK, :);%copy centre by CLK series for rand initial training.
end


