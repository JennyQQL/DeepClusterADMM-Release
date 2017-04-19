function [ pdf ] = gaussianND(X, mu, Sigma)
%GAUSSIANND 
%      X - Matrix of data points, one per row.
%     mu - Row vector for the mean.
%  Sigma - Covariance matrix.

% Get the vector length.
n = size(X, 2);
reg_lambda = 1e-5;
hdim = size(mu,2);
% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian.
pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff * inv(Sigma+reg_lambda*eye(hdim)) .* meanDiff), 2));

end

