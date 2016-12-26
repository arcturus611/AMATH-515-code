%Initialize data
function [A, b, lambda, L, x0] = lasso_init
% small case
% m = 30; n = 128; k = 14;                   % No. of rows (m), columns (n), and nonzeros (k)

% larger case - uncomment to try
m = 300; n = 1280; k = 25;                   % No. of rows (m), columns (n), and nonzeros (k)
[A,Rtmp] = qr(randn(n,m),0);               % Random encoding matrix with orthogonal rows
A  = A';                                   % ... A is m-by-n
p  = randperm(n); p = p(1:k);              % Location of k nonzeros in x
x0 = zeros(n,1); x0(p) = randn(k,1);       % The k-sparse solution
b  = A*x0 + .02*randn(m, 1);               % add random noise   
lambda_max = norm( A'*b, 'inf' );
lambda = 0.1*lambda_max;
L = 1; % lipschitz const 
end