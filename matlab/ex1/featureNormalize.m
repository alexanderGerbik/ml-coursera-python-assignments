function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
m = size(X, 1);
mu = mean(X);
sigma = std(X);
mu_dup = mu(ones(m, 1),:);
sigma_dup = sigma(ones(m, 1),:);
X_norm = (X - mu_dup) ./ sigma_dup;

end
