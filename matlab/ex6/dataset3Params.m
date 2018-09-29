function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
% variants = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
% k = size(variants, 1);
% errors = ones(k, k);
% for i=1:k,
%     for j=1:k,
%         (i-1)*k+j-1, k*k
%         model= svmTrain(X, y, variants(i, 1), @(x1, x2) gaussianKernel(x1, x2, variants(j, 1))); 
%         predictions = svmPredict(model, Xval);
%         errors(i, j) = mean(double(predictions ~= yval));
%     end;
% end;
% [i,j] = find(errors==min(errors(:)));
% C = variants(i,1);
% sigma = variants(j,1);

end
