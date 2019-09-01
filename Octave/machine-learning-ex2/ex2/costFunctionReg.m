function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
cost_matrix = sigmoid(theta' * X');
cost_matrix = log(cost_matrix)*y + log(1-cost_matrix)*(1-y);
J = -cost_matrix/m + (sum(theta.^2) - theta(1)^2)*lambda/(2*m);

grad = ((sigmoid(theta'*X') - y') * X)'/m;
grad(2:end) = grad(2:end) + lambda*theta(2:end)/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
