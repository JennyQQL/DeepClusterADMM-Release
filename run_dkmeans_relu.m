%Run clustering now
function run_dkmeans(datasetname, rho)
% datasetname = 'usps';
% rho = 1;
% use_gpu = false;
cur_path = '.';
data = load([cur_path '\data\' datasetname 'data.mat']); %load data
clusterdata = data.clusterdata;
clustertargets = data.clustertargets;

cluster_staus_path=strcat(cur_path,['\tmp\' datasetname '_km_'  num2str(rho) '.mat']);
load([cur_path '\fineweights\' datasetname '_fine_weights.mat']); % load finetuned weights

num_center = size(clustertargets,2);

%% load pre-trained weights
[basis,centro,RL] = randinitial(clusterdata,clustertargets,w1,w2,w3,w4);
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
R_cluster = 1;

if ~isempty(strfind(datasetname,'reuters'))
    R_cluster = 0.01;
end
R_data = 1;
batchsize = 100;

MAX_ITER = 200;
useAdaDelta = false;
useAdaGrad = false;
finetuninglrate = 1e-3;

if useAdaGrad
    epsilon = 1e-7;
    VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
end
accvalues = zeros(1,MAX_ITER);
nmivalues = zeros(1,MAX_ITER);
purvalues = zeros(1,MAX_ITER);
%% %% if use adadelta%%%%%%%%%%%%%%%
if useAdaDelta
	beta = 0.9;
	VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
	accG = zeros(length(VV),1);
	accD = zeros(length(VV),1);
	epsilon = 1e-7;
end
%% initialize parameters \theta centers and u 
hdim = size(centro,2);
z = netcomput_R(clusterdata,w1,w2,w3,w4); % warm starts
u = zeros(N,hdim)*0.01;
CL = getclusterlabel(clusterdata,centro,w1,w2,w3,w4);
%% begin iteration
best_acc = 0;
best_nmi = 0;
best_pur = 0;
losses = zeros(1,MAX_ITER);
for iter = 1:MAX_ITER
    fprintf(1,'========= runing admm epoch %d ==========>\n',iter);
	% minimize \theta under current centers and u 
	% update \theta
	for epoch = 1:Max_epoch
		tt=0;
        local_f = 0;
        G = zeros(size(VV));
		for batch = 1:N/batchsize
			tt=tt+1;
			cur_data = clusterdata((((tt-1)*batchsize+1):tt*batchsize),:);
            cur_z  = z((((tt-1)*batchsize+1):tt*batchsize),:);
            cur_u  = u((((tt-1)*batchsize+1):tt*batchsize),:);

			VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
			Dim = [l1; l2; l3; l4; l5; l6; l7; l8; l9 ];
            if isempty(strfind(datasetname,'reuters'))
                [f, df] = CG_CLUSTER_ADMM(VV,Dim,cur_data, cur_z, cur_u ,rho,R_data,R_cluster);
            else
                [f, df] = CG_CLUSTER_ADMM_R(VV,Dim,cur_data, cur_z, cur_u ,rho,R_data,R_cluster);
            end
            if isCheckGradient && iter == 1 && epoch ==1 && batch==1
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
        end
        local_f = local_f/batch;
    end
    fprintf('Clustering epoch=%d\t loss =  %.4f ',iter, local_f);
    losses(iter) = local_f;
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
    targetout = netcomput_R(clusterdata,w1,w2,w3,w4);
    z = (basis + rho*(targetout - u))/(1+rho);

    %% update centers
    for j=1:num_center
        centro(j,:) = mean(z(z_label==j,:));
    end
    CL = z_label;
    t_acc = acc(RL,CL);
    t_nmi = nmi(CL',RL');
    t_pur = pur(CL',RL');
    accvalues(iter) = t_acc;
    nmivalues(iter) = t_nmi;
    purvalues(iter) = t_pur;
    if t_acc > best_acc
        best_acc = t_acc;
    end
    if t_nmi > best_nmi
        best_nmi = t_nmi;
    end
    if t_pur > best_pur
        best_pur = t_pur;
    end
    fprintf('\tacc = %.4f\t nmi = %.4f\t pur= %.4f\n',t_acc, t_nmi, t_pur);
	savetxtdata(iter,RL,CL,[cur_path '\tmp\km_' datasetname num2str(rho) '.txt']);%computing NMI,Purity,Accuracy and then saved as '*.txt' file
    save(cluster_staus_path,'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'RL', 'CL', 'iter', 'basis', 'centro');
	%% update u 
    u = u+ (z -targetout);
	
end
plot(losses);



