function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

K = size(centroids, 1);
m = size(X,1);
idx = zeros(m, 1);

for i = 1:m,
    current = X(i, :);
    t = repmat(current, K, 1);
    t = t - centroids;
    t = t .^ 2;
    t = sum(t, 2);
    [val, id] = min(t, [], 1);
    idx(i) = id;
end;

end

