function [C,S,output] = outcomput(digitdata,CL,RL,w1,w2,w3,w4,w5,w6,w7,w8)
%This code is for MNIST clustering. It can compute the cluster centers as well as the distribution of clustering.
N =size(digitdata,1);
targetout = netcomput(digitdata,w1,w2,w3,w4);%get hidden layer value

%get 10 cluster centre 'C' in visual layer(or down layer)
C = zeros(10,10);%C is the cluster center in code layer
counter = zeros(1,10);
for i = 1:N
    C(CL(i),:) = C(CL(i),:)+ targetout(i,:);
    counter(CL(i)) = counter(CL(i))+1;
end

for i = 1:10
    C(i,:)= C(i,:)./ counter(i);
end

%count the distribution information
S = zeros(10,10);
for i = 1:N
    S(CL(i),RL(i)) = S(CL(i),RL(i))+1;
end

%compute the cluster center via the decode nets.
w4probs = [C  ones(10,1)];
w5probs = 1./(1 + exp(-w4probs*w5)); w5probs = [w5probs  ones(10,1)];
w6probs = 1./(1 + exp(-w5probs*w6)); w6probs = [w6probs  ones(10,1)];
w7probs = 1./(1 + exp(-w6probs*w7)); w7probs = [w7probs  ones(10,1)];
output = 1./(1 + exp(-w7probs*w8));

