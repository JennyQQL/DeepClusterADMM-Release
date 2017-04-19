function [batchdata,batchtargets]=data2batch(clusterdata,clustertargets)
%this function can convert data-like(2 dims) into batch-like data(3 dims) one. 
[num,dims]= size(clusterdata);
clusternum = size(clustertargets,2);
num_case=100;%you need to set num of batch here.
batchdata = zeros(num_case,dims,floor(num/num_case));
batchtargets = zeros(num_case,clusternum,floor(num/num_case));
for i = 1:floor(num/num_case)
    batchdata(:,:,i) = clusterdata((i-1)*num_case+1:i*num_case,:);
    batchtargets(:,:,i) = clustertargets((i-1)*num_case+1:i*num_case,:);
end