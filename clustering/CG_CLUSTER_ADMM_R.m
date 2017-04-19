%The CG provider function for clustering
%function [f, df] = CG_CLUSTER_ADMM(VV,Dim,XX, Hcen,centro, CL, NJ, u ,rho, R_data,R_cluster)
function [f, df,f0,f1] = CG_CLUSTER_ADMM(VV,Dim,XX, z, u ,rho, R_data,R_cluster)
% changing log
% substitute cross entropy to mse
l1 = Dim(1);
l2 = Dim(2);
l3 = Dim(3);
l4= Dim(4);
l5= Dim(5);
l6= Dim(6);
l7= Dim(7);
l8= Dim(8);
l9= Dim(9);
N = size(XX,1);
actFunc = @relu;
invAct  = @invRelu;
outActFunc = @linear;
outInvFunc = @invLinear;
% outActFunc = @relu;
% outInvFunc = @invRelu;

% featActFun = @sigmoid;
% featInvFun = @invSigmoid;
featActFun = @linear;
featInvFun = @invLinear;
% featActFun = @relu;
% featInvFun = @invRelu;
% Do decomversion.
 w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2 = reshape(VV(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 xxx = xxx+(l2+1)*l3;
 w3 = reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
 xxx = xxx+(l3+1)*l4;
 w4 = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
 xxx = xxx+(l4+1)*l5;
 w5 = reshape(VV(xxx+1:xxx+(l5+1)*l6),l5+1,l6);
 xxx = xxx+(l5+1)*l6;
 w6 = reshape(VV(xxx+1:xxx+(l6+1)*l7),l6+1,l7);
 xxx = xxx+(l6+1)*l7;
 w7 = reshape(VV(xxx+1:xxx+(l7+1)*l8),l7+1,l8);
 xxx = xxx+(l7+1)*l8;
 w8 = reshape(VV(xxx+1:xxx+(l8+1)*l9),l8+1,l9);
 %get targetout,and reconstructed data
 XX = [XX ones(N,1)];
 w1probs = actFunc(XX*w1); w1probs = [w1probs  ones(N,1)];
 w2probs = actFunc(w1probs*w2); w2probs = [w2probs ones(N,1)];
 w3probs = actFunc(w2probs*w3); w3probs = [w3probs  ones(N,1)];
 w4probs = featActFun(w3probs*w4); 
 targetout = w4probs;
 w4probs = [w4probs  ones(N,1)];
 w5probs = actFunc(w4probs*w5); w5probs = [w5probs  ones(N,1)];
 w6probs = actFunc(w5probs*w6); w6probs = [w6probs  ones(N,1)];
 w7probs = actFunc(w6probs*w7); w7probs = [w7probs  ones(N,1)];
 XXout     = outActFunc(w7probs*w8);

 %f0 = -R_cl uster/N*sum(sum((z+u).*log(targetout)));
 f0 = R_cluster*rho/(2*N)*sum(sum( (z - targetout + u).^2));
 %f0 = rho/(2*N)*sum(sum( (z+u - targetout).^2));
 f1 = R_data/(2*N)*sum(sum( (XX(:,1:end-1)-XXout).^2));
 %f1 = -R_data/N*sum(sum( XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout)));
 f = f0 + f1;

%get gradient of w1--w8
IO = R_data*(XXout-XX(:,1:end-1)).*outInvFunc(XXout);
Ix8=IO; 
dw8 =  w7probs'*Ix8/N;

Ix7 = (Ix8*w8').*invAct(w7probs); 
Ix7 = Ix7(:,1:end-1);
dw7 =  w6probs'*Ix7/N;

Ix6 = (Ix7*w7').*invAct(w6probs); 
Ix6 = Ix6(:,1:end-1);
dw6 =  w5probs'*Ix6/N;

Ix5 = (Ix6*w6').*invAct(w5probs); 
Ix5 = Ix5(:,1:end-1);
dw5 =  w4probs'*Ix5/N;

Ix4 = (Ix5*w5').*featInvFun(w4probs);
Ix4p = R_cluster*rho*(targetout- z-u).*featInvFun(targetout);
Ix4 = Ix4(:,1:end-1) + Ix4p;
% Ix4 = Ix4(:,1:end-1) - R_cluster*rho/N*(z-targetout+u).*featInvFun(targetout);
%Ix4 = Ix4(:,1:end-1)+R_cluster/N*(u+z).*(targetout-1);
dw4 =  w3probs'*Ix4/N;

Ix3 = (Ix4*w4').*invAct(w3probs); 
Ix3 = Ix3(:,1:end-1);
dw3 =  w2probs'*Ix3/N;

Ix2 = (Ix3*w3').*invAct(w2probs); 
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2/N;

Ix1 = (Ix2*w2').*invAct(w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1/N;

df = [dw1(:)' dw2(:)' dw3(:)' dw4(:)' dw5(:)' dw6(:)'  dw7(:)'  dw8(:)'  ]'; 
end
function y = sigmoid(x)
    y = 1./(1+exp(-x));
end

function y = relu(x)
    y = max(0,x);
end
function y = softplus(x)
    y = log(1+exp(x));
end
function y = invSoftplus(x)
    y = (exp(x)-1)./exp(x);
end
function y = invSigmoid(x)
    %y = sigmoid(x);
    y = x.*(1-x);
end
function y = invTanh(x)
    y = 1 - x.^2;
end

function y = invRelu(x) 
    y = double(x>0);
end
function y = linear(x)
 y =x;
end
function dx = invLinear(x)
 dx = ones(size(x));
end