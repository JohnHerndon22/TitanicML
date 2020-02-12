%% Machine Learning Online Class - Exercise 2: Logistic Regression - applied to Titanic

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.

% data = load("new_mtrain.txt");
data = csvread("prep_train.csv");
X = data(:, [4, 8, 15, 18, 19, 20, 21, 22]); 
% keyboard;
lambda = 4;

y = data(:, 3);
% handled by preprocessing
% newX = reg_fare(X);
% X = newX;

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = Titan_costFunction(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);


%% ============= Part 3: Optimizing using fminunc  =============
%  In this exercise, you will use a built-in function (fminunc) to find the
%  optimal parameters theta.

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 1000);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(Titan_costFunction(t, X, y, lambda)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%% ============== Part 4: Predict and Accuracies ==============

% Compute accuracy on our training set
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('\n');

% now for the test dataset
data = csvread("prep_test.csv");
testX = data(:, [3, 7, 14, 17, 18, 19, 20, 21]);

% data(5, [4, 8, 12, 15, 16, 17, 18, 19])
disp('data loaded....')

[m, n] = size(testX);
passNums = data(:,2);

predictions = [zeros(m,1)];

% newX = reg_fare(testX);
% testX = newX;

testX = [ones(m, 1) testX];

% [theta, cost] = fminunc(@(theta)(costFunction(theta, testX, predictions)), initial_theta, options);
predictions = predict(theta, testX);
% keyboard;
testy = [passNums, predictions];
% testy = [['PassengerId', 'Survived']; testy]

disp('predictions made....')

csvwrite('normalize results 01-20 v4.csv', testy);


% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);



