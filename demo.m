% demo to reproduce the results in our paper
% 
addpath(genpath('.'));
%% run on mnist
% dc-kmeans
fprintf('Running dc-kmeans on MNIST dataset\n');
run_dkmeans('mnist_full',1);
% dc-gmm
fprintf('Running dc-gmm on MNIST dataset\n');
run_dgmm_full('mnist_full',1000);

pause
%% run on usps
% dc-kmeans
fprintf('Running dc-kmeans on USPS dataset\n');
run_dkmeans('usps',1);
% dc-gmm
fprintf('Running dc-gmm on USPS dataset\n');
run_dgmm_full('usps',1000);
pause

%% run on Reuters10k
% dc-kmeans
fprintf('Running dc-kmeans on Reuters10k dataset\n');
run_dkmeans('usps',1);
% dc-gmm
fprintf('Running dc-gmm on Reuters10k dataset\n');
run_dgmm_full('usps',1000);