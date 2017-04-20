function out = netcomput(data,w1,w2,w3,w4)
%this function can get top(or hidden)layer value with autoencode network, with sigmoid function.
N =size(data,1);
data = [data ones(N,1)];
actFun = @relu;
 featFun = @linear;
% featFun = @relu;
w1probs = actFun(data*w1); w1probs = [w1probs  ones(N,1)];
w2probs = actFun(w1probs*w2); w2probs = [w2probs ones(N,1)];
w3probs = actFun(w2probs*w3); w3probs = [w3probs  ones(N,1)];
out = featFun(w3probs*w4);
end
function y = linear(x)
 y =x;
end

function y = relu(x)
    y = max(0,x);
end

function y = sigmoid(x)
    y = 1./(1+exp(-x));
end