function [pSigma,init_Priors] = initializeGMM_full(targetout,CL,centro)
   
    [N,hdim] = size(targetout);
    num_center = size(centro,1);
   % Run Kmeans to pre-cluster the data
   [assignments, initMeans] = kmeans(targetout,num_center,'MaxIter',50);
   
    % pSigma
    pSigma = zeros(hdim,hdim,num_center);
    init_Priors = zeros(1,num_center);
    %Ncount = zeros(1,num_center);
    for i=1:num_center
        data_k = targetout(assignments == i,:);
        init_Priors(i) = size(data_k,1)/N;
        if size(data_k,1)==0 || size(data_k,2)==0
            pSigma(:,:,i) = diag(diag(cov(targetout)));
        else
            pSigma(:,:,i) = diag(diag(cov(data_k)));
        end
        %pSigma(:,:,i) = cov(targetout(CL==i,:));
        %Ncount(i) = sum(CL==i);
    end
    %pi = Ncount/N;
   
end