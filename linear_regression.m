% Clear variables and a screen.
clear; 
close all; 
clc;

% Loading training data from file ----------------------------------------------
fprintf('Loading the training data from file...\n\n');

% Loading training data from file.
data = load('student_score.csv');

% Split data into features and results.
X = data(:, 1:2);
y = data(:, 3);

% COST function.
% It shows how accurate our model is based on current model parameters.
function [cost] = cost_function(X, y, theta, lambda)
    % Input:
    % X - input features - (m x n) matrix.
    % theta - our model parameters - (n x 1) vector.
    % y - a vector of correct output - (m x 1) vector.
    % lambda - regularization parameter.
    %
    % Output:
    % cost - number that represents the cost (error) of our model with specified parameters theta.
    %
    % Where:
    % m - number of training examples,
    % n - number of features.

    % Get the size of the trainging set.
    m = size(X, 1);

    % Get the difference between predictions and correct output values.
    differences = hypothesis(X, theta) - y;

    % Calculate regularization parameter.
    % Remember that we should not regularize the parameter theta_zero.
    theta_cut = theta(2:end, 1);
    regularization_param = lambda * (theta_cut' * theta_cut);

    % Calculate current predictions cost.
    cost = (1 / 2 * m) * (differences' * differences + regularization_param);
end

% FEATURE NORMALIZE function.
% Normalizes the features in X. Returns a normalized version of X where the mean value of 
% each feature is 0 and the standard deviation is 1.
function [X_normalized, mu, sigma] = feature_normalize(X)
    X_normalized = X;
    mu = zeros(1, size(X_normalized, 2));
    sigma = zeros(1, size(X_normalized, 2));

    % Get average values for each feature (column) in X.
    mu = mean(X_normalized);

    % Calculate the standard deviation for each feature.
    sigma = std(X_normalized);

    % Subtract mean values from each feature (column) of every example (row)
    % to make all features be spread around zero.
    X_normalized = X_normalized - mu;

    % Normalize each feature values for each example so that all features 
    % are close to [-1:1] boundaries.
    X_normalized = X_normalized ./ sigma;
end

% GRADIENT DESCENT function.
% It calculates what steps (deltas) should be taken for each theta parameter in
% order to minimize the cost function.
function [theta, J_history] = gradient_descent(X, y, theta, alpha, lambda, num_iterations)
    % Input:
    % X - training set of features - (m x n) matrix.
    % y - a vector of expected output values - (m x 1) vector.
    % theta - current model parameters - (n x 1) vector.
    % alpha - learning rate, the size of gradient step we need to take on each iteration.
    % lambda - regularization parameter.
    % numb_iterations - number of iterations we will take for gradient descent.
    % 
    % Output:
    % theta - optimized theta parameters - (m x 1) vector.
    % J_history - the history cost function changes over iterations.
    %
    % Where:
    % m - number of training examples,
    % n - number of features.

    % Get number of training examples.
    m = size(X, 1);

    % Initialize J_history with zeros.
    J_history = zeros(num_iterations, 1);

    for iteration = 1:num_iterations
        % Perform a single gradient step on the parameter vector theta.
        theta = gradient_step(X, y, theta, alpha, lambda);

        % Save the cost J in every iteration.  
        J_history(iteration) = cost_function(X, y, theta, lambda);
    end
end

% GRADIENT STEP function.
% Function performs one step of gradient descent for theta parameters.
function [theta] = gradient_step(X, y, theta, alpha, lambda)
    % Input:
    % X - training set of features - (m x n) matrix.
    % y - a vector of expected output values - (m x 1) vector.
    % theta - current model parameters - (n x 1) vector.
    % alpha - learning rate, the size of gradient step we need to take on each iteration.
    % lambda - regularization parameter.
    %
    % Output:
    % theta - optimized theta parameters - (m x 1) vector.
    % J_history - the history cost function changes over iterations.
    %
    % Where:
    % m - number of training examples,
    % n - number of features.

    % Get number of training examples.
    m = size(X, 1);

    % Predictions of hypothesis on all m examples.
    predictions = hypothesis(X, theta);

    % The difference between predictions and actual values for all m examples.
    difference = predictions - y;

    % Calculate regularization parameter.
    regularization_param = 1 - alpha * lambda / m;

    % Vectorized version of gradient descent.
    theta = theta * regularization_param - alpha * (1 / m) * (difference' * X)';

    % We should NOT regularize the parameter theta_zero.
    theta(1) = theta(1) - alpha * (1 / m) * (X(:, 1)' * difference)';
end

% HYPOTHESIS function.
% It predicts the output values y based on the input values X and model parameters.
function [predictions] = hypothesis(X, theta)
    % Input:
    % X - input features - (m x n) matrix.
    % theta - our model parameters - (n x 1) vector.
    %
    % Output:
    % predictions - output values that a calculated based on model parameters - (m x 1) vector.
    %
    % Where:
    % m - number of training examples,
    % n - number of features.

    predictions = X * theta;
end

% LINEAR REGRESSION function.
function [theta mu sigma X_normalized J_history] = linear_regression_train(X, y, alpha, lambda, num_iterations)
    % X - training set.
    % y - training set output values.
    % alpha - learning rate (gradient descent step size).
    % lambda - regularization parameter.
    % num_iterations - number of gradient descent steps.

    % Calculate the number of training examples.
    m = size(y, 1);

    % Calculate the number of features.
    n = size(X, 2);

    % Normalize features.
    [X_normalized mu sigma] = feature_normalize(X);

    % Add a column of ones to X.
    X_normalized = [ones(m, 1), X_normalized];

    % Initialize model parameters.
    initial_theta = zeros(n + 1, 1);

    % Run gradient descent.
    [theta, J_history] = gradient_descent(X_normalized, y, initial_theta, alpha, lambda, num_iterations);
end

% NORMAL EQUATION function.
% Closed-form solution to linear regression.
function [theta] = normal_equation(X, y)
    theta = pinv(X' * X) * X' * y;
end




% Plotting training data -------------------------------------------------------
fprintf('Plotting the training data...\n\n');

% Split the figure on 2x2 sectors.
% Start drawing in first sector.
subplot(2, 2, 1);

scatter3(X(:, 1), X(:, 2), y, [], y(:), 'o');
title('Training Set');
xlabel('Study');
ylabel('Sleep');
zlabel('Score');

% Running linear regression ----------------------------------------------------
fprintf('Running linear regression...\n');

% Setup regularization parameter.
lambda = 0;

alpha = 0.1;
num_iterations = 50;
[theta mu sigma X_normalized J_history] = linear_regression_train(X, y, alpha, lambda, num_iterations);

fprintf('- Initial cost: %f\n', J_history(1));
fprintf('- Optimized cost: %f\n', J_history(end));

fprintf('- Theta (with normalization):\n');
fprintf('-- %f\n', theta);
fprintf('\n');

% Calculate model parameters using normal equation -----------------------------
fprintf('Calculate model parameters using normal equation...\n');

X_normal = [ones(size(X, 1), 1) X];
theta_normal = normal_equation(X_normal, y);
normal_cost = cost_function(X_normal, y, theta_normal, lambda);

fprintf('- Normal function cost: %f\n', normal_cost);

fprintf('- Theta (without normalization):\n');
fprintf('-- %f\n', theta_normal);
fprintf('\n');

% Plotting normalized training data --------------------------------------------
fprintf('Plotting normalized training data...\n\n');

% Start drawing in second sector.
subplot(2, 2, 2);

scatter3(X_normalized(:, 2), X_normalized(:, 3), y, [], y(:), 'o');
title('Normalized Training Set');
xlabel('Normalized Study');
ylabel('Normalized Sleep');
zlabel('Score');

% Draw gradient descent progress ------------------------------------------------
fprintf('Plot gradient descent progress...\n\n');

% Continue plotting to the right area.
subplot(2, 2, 3);

plot(1:num_iterations, J_history);
xlabel('Iteration');
ylabel('J(\theta)');
title('Gradient Descent Progress');

% Plotting hypothesis plane on top of training set -----------------------------
fprintf('Plotting hypothesis plane on top of training set...\n\n');

% Get apartment study hours and sleep hours boundaries.
apt_study = X_normalized(:, 2);
apt_sleep = X_normalized(:, 3);
apt_study_range = linspace(min(apt_study), max(apt_study), 10);
apt_sleep_range = linspace(min(apt_sleep), max(apt_sleep), 10);

% Calculate predictions for each possible combination of sleeping hours number and appartment size.
apt_score = zeros(length(apt_study_range), length(apt_sleep_range));
for apt_study_index = 1:length(apt_study_range)
    for apt_sleep_index = 1:length(apt_sleep_range)
        X = [1, apt_study_range(apt_study_index), apt_sleep_range(apt_sleep_index)];
        apt_score(apt_study_index, apt_sleep_index) = hypothesis(X, theta);
    end
end

% Plot the plane on top of training data to see how it feets them.
subplot(2, 2, 2);
hold on;
mesh(apt_study_range, apt_sleep_range, apt_score);
legend('Training Examples', 'Hypothesis Plane')
hold off;
