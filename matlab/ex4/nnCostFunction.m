function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

yv = zeros(m, num_labels);
for i = 1:m,
    yv(i,y(i)) = 1;
end
y = yv;

a1 = X;
a1 = [ones(size(a1, 1), 1) a1];
z2 = a1 * Theta1';
a2 = sigmoid(z2);

a2 = [ones(size(a2, 1), 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;
J = -sum(sum(y .* log(h) + (1-y) .* log(1-h))) / m;
J = J + (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)))* lambda / 2 / m;

% Part 2: Implement the backpropagation algorithm to compute the gradients
delta3 = a3 - y;
delta2 = delta3 * Theta2;
delta2 = delta2(:, 2:end);
delta2 = delta2 .* sigmoidGradient(z2);

Theta1_grad = delta2' * a1 / m;
Theta2_grad = delta3' * a2 / m;


% Part 3: Implement regularization with the cost function and gradients.
t1 = Theta1 * lambda / m;
t1(:, 1) = zeros(size(Theta1, 1), 1);
Theta1_grad = Theta1_grad + t1;

t2 = Theta2 * lambda / m;
t2(:, 1) = zeros(size(Theta2, 1), 1);
Theta2_grad = Theta2_grad + t2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
