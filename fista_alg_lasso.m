function output = fista_alg_lasso(A, b, lambda)%% Implement accelerated splitting here
% pre-calculate stuff
AtA = A'*A;
Atb = A'*b; 
sz = size(A);
n = sz(2);

% initialize variables
L = 1; 
x_fista_curr = zeros(n, 1); 
x_fista_prev_1 = zeros(n, 1); 
x_fista_prev_2 = zeros(n, 1);
delta = 10;
epsilon = 10^(-8);
nIter = 1; 

while(delta > epsilon)
    y = x_fista_prev_1 + (nIter/(nIter+3))*(x_fista_prev_1 - x_fista_prev_2);
    grad_y  = grad_f(y, AtA, Atb);
    x_fista_curr = prox_op_l1(y-(1/L)*grad_y, lambda/L);
    
    delta_vec_fista(nIter) = norm(x_fista_curr - x_fista_prev_1, 2);
    delta = delta_vec_fista(nIter);
    
    x_fista_prev_2 = x_fista_prev_1;
    x_fista_prev_1 = x_fista_curr;
    nIter = nIter + 1; 
end

output.x_opt = x_fista_curr;
output.nIter = nIter;
output.delta_vec = delta_vec_fista;
end