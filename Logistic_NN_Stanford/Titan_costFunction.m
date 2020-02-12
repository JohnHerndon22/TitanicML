function [J, grad] = Titan_costFunction(theta, X, y, lambda)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
newtheta = theta;
newtheta(1,:) = [];

hypoth = X*theta;
newhy = sigmoid(hypoth);
regex1 = lambda/(2*m); 
regex2 = newtheta'*newtheta;   % need theta with out the first element           

J = (((-y'*log(newhy)) - ((1-y)'*log(1-newhy)))*(1/m))+(regex1*regex2);              % regex1 will go onto the end of this function


% this is from the old function
errors = newhy - y;
regex1a = lambda / m;
lamelement = regex1a * newtheta;
lamelement = [0;lamelement];
grad = ((1/m) * (X'*errors)) + lamelement;








% =============================================================

end
