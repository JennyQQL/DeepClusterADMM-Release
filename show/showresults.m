% show clustering result of mnist
data_path =  strcat(cur_path,'\data\mnistdata.mat');
cluster_path = strcat(cur_path,'\tmp\cluster_status.mat');
load(data_path);
load(cluster_path);

[C,S,output] = outcomput(clusterdata,CL,RL,w1,w2,w3,w4,w5,w6,w7,w8);
%show 10 clustering centre
figure('Position',[100,600,1000,200]);
mnistdisp(output');
hold on;

%show distribution of the clustering results
figure(2)
X = 0:9;
for i = 1:5
    subplot(4,5,i);
    bar(X,S(i,:));
end
for i = 6:10
    subplot(4,5,i);
    mnistdisp(output(i-5,:)');
end
for i = 11:15
    subplot(4,5,i);
    bar(X,S(i-5,:));
end
for i = 16:20
    subplot(4,5,i);
    mnistdisp(output(i-10,:)');
end
hold on;

