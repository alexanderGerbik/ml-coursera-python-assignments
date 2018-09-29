function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

m = size(X, 1);
X_poly = zeros(numel(X), p);
for i = 1:m,
    t = zeros(1, p);
    for j = 1:p,
        t(1, j) = X(i, 1) ^ j;
    end;
    X_poly(i, :) = t;
end;

end
