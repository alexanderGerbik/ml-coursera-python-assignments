function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y);
h = X * theta;
diff = h - y;
th = theta(2:end, :);
J = diff' * diff / 2 / m + th' * th * lambda / 2 / m;
grad = ((h - y)' * X / m)';
reg = theta * lambda / m;
reg(1, 1) = 0;
grad = grad + reg;

grad = grad(:);

end
