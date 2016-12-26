% AMATH 515 Homework 1 Starter

clear all; close all; clc

%set up data
%rand('twister',0); randn('state',0);

% small case
m = 30; n = 128; k = 14;                   % No. of rows (m), columns (n), and nonzeros (k)

% larger case - uncomment to try
% m = 300; n = 1280; k = 50;                   % No. of rows (m), columns (n), and nonzeros (k)

[A,Rtmp] = qr(randn(n,m),0);               % Random encoding matrix with orthogonal rows
A  = A';                                   % ... A is m-by-n
p  = randperm(n); p = p(1:k);              % Location of k nonzeros in x
x0 = zeros(n,1); x0(p) = randn(k,1);       % The k-sparse solution
b  = A*x0 + .02*randn(m, 1);               % add random noise   

% plot(x0)

lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;
% 
%% Implement coordinate descent here
%% Initialize x^(0) = 0
x_curr = zeros(n, 1); 
x_prev = zeros(n, 1);
epsilon = 10^(-8);
delta = 10;

while (delta > epsilon)
    x_prev = x_curr;
    for i = 1:n %step through all the coordinates
        %calculate residual 
        x_rem = x_curr;
        x_rem(i) = 0;
        res = b - A*x_rem;
        
        %update ith coordinate
        temp = norm(A(:, i), 2)^2;
        x_curr(i) = prox_op_l1(A(:, i)'*res/temp, lambda/temp);
    end
    delta = norm(x_curr - x_prev, 2);
end

xCoord = x_curr;

%% Implement proximal splitting here
%calculate constants
grad_t1 = A'*A; 
grad_t2 = A'*b;
L = 1; %arbitrary, I don't understand how it is supposed to be chosen
x_prox_grad_curr = zeros(n, 1);
x_prox_grad_prev = zeros(n, 1);
delta = 10;
epsilon = 10^(-8);
iter = 1;
while (delta > epsilon)
    x_prox_grad_prev = x_prox_grad_curr;
    grad_x = grad_f(x_prox_grad_prev, grad_t1, grad_t2);
    x_prox_grad_curr = prox_op_l1(x_prox_grad_prev - (1/L)*grad_x, lambda*(1/L)); %wasn't converging well without lambda, and I don't get why this works
    delta_vec(iter) = norm(x_prox_grad_curr - x_prox_grad_prev, 2);
    delta = delta_vec(iter);
    iter = iter+1;
end

xProx = x_prox_grad_curr;

%% Implement accelerated splitting here
grad_t1 = A'*A;
grad_t2 = A'*b; 
L = 1; 
x_fista_curr = zeros(n, 1); 
x_fista_prev_1 = zeros(n, 1); 
x_fista_prev_2 = zeros(n, 1);
delta = 10;
epsilon = 10^(-8);
iter_fista = 1; 
while(delta > epsilon)
    y = x_fista_prev_1 + (iter/(iter+3))*(x_fista_prev_1 - x_fista_prev_2);
    grad_y  = grad_f(y, grad_t1, grad_t2);
    x_fista_curr = prox_op_l1(y-(1/L)*grad_y, lambda/L);
    
    delta_vec_fista(iter_fista) = norm(x_fista_curr - x_fista_prev_1, 2);
    delta = delta_vec_fista(iter_fista);
    
    x_fista_prev_2 = x_fista_prev_1;
    x_fista_prev_1 = x_fista_curr;
    iter_fista = iter_fista + 1; 
end

xFastProx = x_fista_curr;

%% Run CVX here 
cvx_begin quiet
    cvx_precision low
    variable xCVX(n)
    minimize 0.5*sum_square(A*xCVX - b) + lambda*norm(xCVX, 1)
cvx_end

%% Uncomment lines to see how close your answer is


fprintf('\n Distance between CVX solution and coordinate descent: %5.4f \n', norm(xCoord - xCVX));
fprintf('\n Distance between CVX solution and proximal splitting: %5.4f \n', norm(xProx - xCVX));
fprintf('\n Distance between CVX solution and proximal splitting: %5.4f \n', norm(xFastProx - xCVX));
