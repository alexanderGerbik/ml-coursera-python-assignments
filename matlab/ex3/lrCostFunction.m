function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
h = sigmoid(X * theta);
J = -(y' * log(h) + (1-y') * log(1-h)) / m + sum(theta(2:end, :) .^ 2) * lambda / 2 / m;
grad = ((h - y)' * X / m)';
reg = theta * lambda / m;
reg(1, 1) = 0;
grad = grad + reg;

end
