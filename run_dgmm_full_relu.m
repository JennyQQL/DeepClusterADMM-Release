%Run GMM clustering now
function run_dgmm_full(datasetname, rho)
% datasetname = 'mnist_full';
% rho = 1000;

cur_path = '.';
data = load([cur_path '\data\' datasetname 'data.mat']); %load data
clusterdata = data.clusterdata;
clustertargets = data.clustertargets;

%randseed(1);
Max_epoch = 1;
num_center = size(clustertargets,2);
batchsize = 100;
reg_lambda = 1e-5;
PI = 3.1415926;
R_data = 1;

%% initialization
N = size(clusterdata,1);

%% Hyper parameters setting

MAX_ITER = 300;
useAdaDelta = false;
useAdaGrad = false;

accvalues = zeros(1,MAX_ITER);
nmivalues = zeros(1,MAX_ITER);
purvalues = zeros(1,MAX_ITER);

finetuninglrate = 5e-4;
R_cluster = 1e-5;

load([cur_path '\fineweights\' datasetname '_fine_weights.mat']); % load finetuned weights
l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w5,1)-1;
l6=size(w6,1)-1;
l7=size(w7,1)-1;
l8=size(w8,1)-1;
l9=l1;
cluster_staus_path=strcat(cur_path,['\tmp\gmm' datasetname num2str(rho) '.mat']);
if useAdaGrad
    epsilon = 1e-7;
    VV = [w1(:)' w2(:)' w3(:)' w4(:)' w5(:)' w6(:)' w7(:)' w8(:)']';
end
%% load pre-trained weights
if ~exist(cluster_staus_path, 'file')
    epochnow=1;
    [basis,centro,RL] = randinitial(clusterdata,clustertargets,w1,w2,w3,w4);
else
    load(cluster_staus_path);
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
hdim = size(centro,2);
z = netcomput_R(clusterdata,w1,w2,w3,w4); % warm starts
u = zeros(N,hdim)*0.01;
CL = getclusterlabel(clusterdata,centro,w1,w2,w3,w4);

% initialize gmm parameters with k-means
fprintf('initializing GMM parameters\n');
targetout = netcomput_R(clusterdata,w1,w2,w3,w4);
[pSigma,pi] = initializeGMM_full(targetout,CL,centro);
%% begin iteration
best_acc = 0;
best_nmi = 0;
best_pur = 0;
    for iter = 1:MAX_ITER
        fprintf(1,'========= runing admm epoch %d ==========>\n',iter);
        % minimize \theta under current centers and u 
        % update \theta
        tic
        for epoch = 1:Max_epoch
            tt=0;
            local_f = 0;
            local_f0 = 0;
            local_f1 = 0;
            G = zeros(size(VV));
            for batch = 1:N/batchsize
                %fprintf(1,'Clustering epoch %d batch %d\r',epoch,batch);
                % assign each epoch with 1000 batches %
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
                local_f = local_f + f;
                local_f0 = local_f0 + f0;
                local_f1 = local_f1 + f1;
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
            
            local_f = local_f/batch;
            
        end
        fprintf('Clustering epoch =  %d loss =  %.4f ',iter,local_f);
        %% update z(dummy variables)
        % get target out
        %zold = z;
        targetout = netcomput_R(clusterdata,w1,w2,w3,w4);
        % compute Px
        Px = zeros(N, num_center);  
        if sum(isnan(centro))>=1
            warnning('NaN occur\n');
            break;
        end
        % For each cluster...
        for k = 1 : num_center
            % Evaluate the Gaussian for all data points for cluster 'j'.
            %Px(:, k) = gaussianND(targetout, centro(k, :), pSigma(:,:,k));
            % 1.22 revised
            Px(:, k) = gaussianND(z, centro(k, :), pSigma(:,:,k));
        end
        
        % compute gamma
        tmp = bsxfun(@times,Px,pi);
        gamma = bsxfun(@rdivide,tmp,sum(tmp,2));
        %% compute z
        invSigma = zeros(size(pSigma));
        for i=1:num_center
            invSigma(:,:,i) = inv(pSigma(:,:,i)+reg_lambda*eye(hdim));
        end
        for i=1:N
            left_coeff_matrix = zeros(hdim,hdim);
            right_side = rho*(targetout(i,:)-u(i,:));
            for k=1:num_center
                left_coeff_matrix = left_coeff_matrix + gamma(i,k)*invSigma(:,:,k);
                right_side = right_side + (gamma(i,k) * centro(k,:)*invSigma(:,:,k));
            end
            left_coeff_matrix = left_coeff_matrix + rho*eye(hdim);
            z(i,:) = (right_side*inv(left_coeff_matrix));
        end
        %% update centers (\mu)
        Ncount = sum(gamma,1);

        [~,CL] = max(gamma,[],2);
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
        fprintf('\tacc=%.4f\t nmi=%.4f\t pur=%.4f\n',t_acc, t_nmi, t_pur);
        savetxtdata(iter,RL,CL,[cur_path '\tmp\gmm' datasetname num2str(rho) '.txt']);%computing NMI,Purity,Accuracy and then saved as '*.txt' file
        save(cluster_staus_path,'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8', 'RL', 'CL', 'iter', 'basis', 'centro','epochnow','gamma');
        %% update gmm parameters
        % mu
        centro = diag(1./Ncount) * gamma' * z;

        for i=1:num_center
            tmp = z - repmat(centro(i,:),N,1);
            pSigma(:,:,i) = tmp' * bsxfun(@times,tmp,gamma(:,i))/Ncount(i);
        end
        % check for convergence
        L = -sum(log(Px * pi'))/N; 
        %fprintf('\tGMM loss %.4f\n',L);
        
        pi = Ncount/N;
        %% update u 
        u = u + 0.5*(z-targetout);

        toc;
    end
    %save(['index' datasetname '.mat'], 'accvalues','nmivalues','purvalues');
%end
fprintf('bset_acc %.4f\t bset_nmi %.4f\t bset_pur %.4f\n',t_acc, t_nmi, t_pur);



