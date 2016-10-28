function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

m = size(X,1);
n = size(theta);
h = X * theta;
hy = (h-y).*(h-y);
thetasq = theta(2:end).*theta(2:end);
J = (1/(2*m))*sum(hy)+(lambda/(2*m))*sum(thetasq);
thetaone = (h-y).*X(:,1);
grad(1) = (1/m)*sum(thetaone);
thetaj = (h-y).*X(1:m,2:n);
grad(2:n) = (1/m)*sum(thetaj)'+(lambda/m)*theta(2:n);





% =========================================================================

grad = grad(:);

end
