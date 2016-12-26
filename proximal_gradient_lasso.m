function output = proximal_gradient_lasso(A, b, lambda, L) 
%% Prox-Gradient algorithm
% pre-calculate constants
AtA = A'*A; 
Atb = A'*b;
delta = 10;
epsilon = 10^(-8);
iter = 1;
prox_thres =  lambda*(1/L);
% allocate memory to vectors
sz = size(A); 
n = sz(2);
x_curr = zeros(n, 1);

%% Iterate until convergence
while (delta > epsilon)
    val(iter) = eval_lasso(A, b, lambda, x_curr); 
    % gradient step
    x_prev = x_curr;
    grad_x = grad_f(x_prev, AtA, Atb);
    
    % prox step
    x_curr = prox_op_l1(x_prev - (1/L)*grad_x, prox_thres); 
    
    % check convergence
    delta_vec(iter) = norm(x_curr - x_prev, 2)^2;
    delta = delta_vec(iter);
    
    iter = iter+1;
end

% pack all output in struct
output.x_opt = x_curr;
output.nIter = iter;
output.delta_vec = delta_vec;
output.val = val; 
end
