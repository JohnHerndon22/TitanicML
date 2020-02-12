function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));
[m, width] = size(z);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).
for count_m = 1:m
    for count_w = 1:width
        g(count_m, count_w) = (1/(1+exp(-z(count_m, count_w)))) * (1-((1/(1+exp(-z(count_m, count_w))))));
    end
end 










% =============================================================




end
