%Run clustering now
% clear
% clc
cur_path = '.';
datasetname = 'usps';
cluster_staus_path=strcat(cur_path,['\tmp\cluster_status' datasetname '_121.mat']);

load(['E:\2017IJCAI\DeepClusteringADMM\tmp\' datasetname '_fine_weights.mat']);
load(['E:\2017IJCAI\DeepClusteringADMM\data\' datasetname 'data.mat']);
clusternum = 10;

%% load pre-trained weights
if ~exist(cluster_staus_path, 'file')
    epochnow=1;
    [basis,centro,RL] = randinitial(clusterdata,clustertargets,w1,w2,w3,w4);
else
    load(cluster_staus_path);
end
%% initialization
N = size(clusterdata,1);
l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w5,1)-1;
l6=size(w6,1)-1;
l7=size(w7,1)-1;
l8=size(w8,1)-1;
l9=l1;
%% Hyper parameters setting
isCheckGradient = false;

Max_epoch = 1;
num_center = 10;
rho = 150;
R_cluster = 1;
R_data = 1;
batchsize = 100;

finetuninglrate = 0.1;
QUIET    = 0;
MAX_ITER = 100;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
useAdaDelta = true;
useAdaGrad = false;
if useAdaGrad
    epsilon = 1e-7;
    VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
end
%% %% if use adadelta%%%%%%%%%%%%%%%
if useAdaDelta
	beta = 0.9;
	VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
	accG = zeros(length(VV),1);
	accD = zeros(length(VV),1);
	epsilon = 1e-7;
end
%% initialize parameters \theta centers and u 
%centro = centro;
hdim = size(centro,2);
%z = zeros(N,hdim);
z = netcomput(clusterdata,w1,w2,w3,w4); % warm starts
u = zeros(N,hdim)*0.01;
CL = getclusterlabel(clusterdata,centro,w1,w2,w3,w4);
NJ = zeros(1,num_center);
for j=1:num_center
		idx = find(CL == j);
        NJ(j) = numel(idx);
end
%% begin iteration
for iter = 1:MAX_ITER
    fprintf(1,'========= runing admm epoch %d ==========>\r',iter);
	% minimize \theta under current centers and u 
	% update \theta
    tic
	for epoch = 1:Max_epoch
		tt=0;
        local_f = 0;
        G = zeros(size(VV));
		for batch = 1:N/batchsize
			%fprintf(1,'Clustering epoch %d batch %d\r',epoch,batch);
			% assign each epoch with 1000 batches %
			tt=tt+1;
			cur_data=[];
			cur_basis = [];
			cur_data = clusterdata((((tt-1)*batchsize+1):tt*batchsize),:);
			cur_basis = basis((((tt-1)*batchsize+1):tt*batchsize),:);    
            cur_cl = CL((((tt-1)*batchsize+1):tt*batchsize));
            cur_z  = z((((tt-1)*batchsize+1):tt*batchsize),:);
            cur_u  = u((((tt-1)*batchsize+1):tt*batchsize),:);
			% Perform CG  with 3 linearsearch %
			max_iter=3;
			VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
			Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9 ];
		%	[X, fX] = minimize(VV,'CG_CLUSTER_ADMM',max_iter,Dim,cur_data, cur_z, cur_u ,rho,R_data,R_cluster);   
            [f, df] = CG_CLUSTER_ADMM(VV,Dim,cur_data, cur_z, cur_u ,rho,R_data,R_cluster); 
            if isCheckGradient
                gradientcheck(VV,Dim,cur_data(1:10,:), cur_z(1:10,:), cur_u(1:10,:) ,rho,R_data,R_cluster);
            end
            
            local_f = local_f + f;
            %df = df + 2e-3*df;
			if useAdaDelta
                accG = beta.* accG + (1-beta).*(df.^2);
                dCurr = -(sqrt(accD + epsilon)./sqrt(accG+epsilon)).*df;
                % update accumulated updates (delta)
                accD = beta.*accD + (1-beta).*(dCurr.^2);
                X = VV + dCurr;
                if sum(isnan(X))
                    disp('nan accur!\n');
                    break;
                end
            elseif useAdaGrad
                G = G + df;
                X = VV - finetuninglrate ./sqrt(G+epsilon).*df;
            else
                
                X = VV - finetuninglrate * df;
            end
            % updata cluster weights %
			w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);
			xxx = (l1+1)*l2;
			w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
			xxx = xxx+(l2+1)*l3;
			w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
			xxx = xxx+(l3+1)*l4;
			w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
			xxx = xxx+(l4+1)*l5;
			w5 = reshape(X(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
			xxx = xxx+(l5+1)*l6;
			w6 = reshape(X(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
			xxx = xxx+(l6+1)*l7;
			w7 = reshape(X(xxx+1:xxx+(l7+1)*l8),l7+1,l8);
			xxx = xxx+(l7+1)*l8;
			w8 = reshape(X(xxx+1:xxx+(l8+1)*l9),l8+1,l9);
            %fprintf(1,'\tClustering epoch %d batch %d loss = %.4f\r',epoch,batch,f);
		end
		%epochnow = epoch+1;
        local_f = local_f/batch;
        fprintf('\tClustering epoch =  %d loss =  %.4f\r',epoch,local_f);
    end
    %% update z
    z_label = zeros(N,1);
    basis = zeros(N,hdim);
    for i=1:N
        dist = zeros(1,num_center);
        for j=1:num_center
            dist(j) = norm(z(i,:)-centro(j,:));
        end
        [~,z_label(i)] = min(dist);
        basis(i,:) = centro(z_label(i),:);
    end
    % get target out
    targetout = netcomput(clusterdata,w1,w2,w3,w4);
    z = (basis + rho*(targetout - u))/(1+rho);

    %% update centers
	centro_old = centro;
% 	[CL, targetout] = getclusterlabel(clusterdata,centro,w1,w2,w3,w4);%get clustering label
% 	for j =1: num_center
% 		idx = find(CL==j);
% 		nj = numel(idx);
%         if nj >0 
%             centro(j,:) = (nj+rho*N)/((nj+rho*N)*nj)*sum(targetout(idx,:),1) - (N*rho)/(N*rho+nj)*u(j,:);
%         else % in case some cluster with nothing
%             warning('bad clustering result, some cluster is empty!');
%             randidx = randi(N,1);
%             centro(j,:) = targetout(randidx,:);
%         end
% 	end
    for j=1:num_center
        idx = find(z_label==j);
        centro(j,:) = mean(z(idx,:));
    end
    CL = z_label;
    %[CL, targetout] = getclusterlabel(clusterdata,centro,w1,w2,w3,w4);
    
	savetxtdata(iter,RL,CL,[cur_path '\' datasetname 'kmeans_mse.txt']);%computing NMI,Purity,Accuracy and then saved as '*.txt' file
	%[CL] = getclusterlabel(clusterdata,centro,w1,w2,w3,w4);%update clustering label
% 	basis = zeros(N,size(targetout,2));
% 	NJ = zeros(1,num_center);
% 	cluster_assigned = cell(1,num_center);
% 	
%     target_mean = zeros(size(centro));
% 	for j=1:num_center
% 		idx = find(CL == j);
% 		cluster_assigned{j} = idx;
%         NJ(j) = numel(idx);
%         if ~isempty(idx)
%             target_mean(j,:) = mean(targetout(idx,:));
%             basis(idx,:) = ones(NJ(j),1)*centro(j,:);
%         end
% 	end
	save(cluster_staus_path,'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'RL', 'CL', 'iter', 'basis', 'centro');
	%% update u 
    u = u+ (z -targetout);
% 	for j=1:num_center
% 		u(j,:) = u(j,:) +target_mean(j,:) - centro(j,:);
% 	end
	% diagnostics, reporting, termination checks
	history.objval(iter) = objective(clusterdata,centro);
	
	history.r_norm(iter) =  norm(targetout - z);
	history.s_norm(iter) = norm((centro - centro_old));
	
	history.eps_pri(iter) = sqrt(N)*ABSTOL; %+ RELTOL*max(norm(target_mean), norm(-centro));
    history.eps_dual(iter)= sqrt(N)*ABSTOL + RELTOL*norm(rho*u);

	if ~QUIET 
        fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
        'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
		fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', iter, ...
            history.r_norm(iter), history.eps_pri(iter), ...
            history.s_norm(iter), history.eps_dual(iter), history.objval(iter));
    end

%     if (history.r_norm(iter) < history.eps_pri(iter) && ...
%        history.s_norm(iter) < history.eps_dual(iter))
%          break;
%     end

    toc;
end
	



