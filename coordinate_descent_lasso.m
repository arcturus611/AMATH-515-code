function cd_lasso = coordinate_descent_lasso(A, b, lambda)
%% coordinate descent here
%% Initialize x^(0) = 0
sz = size(A);
n = sz(2); 
x_curr = zeros(n, 1); 
epsilon = 10^(-8);
delta = 10;
nIter = 0;

%% Pre-calculate some constants that are used repeatedly
col_norm_sqr = diag(A'*A);
prox_thres = (lambda*ones(n, 1))./col_norm_sqr;

%% loop through each coordinate update until convergence
while (delta > epsilon)
    nIter = nIter + 1;
    
    x_prev = x_curr;
    for i = 1:n %step through each coordinate
        %update residual 
        x_rem = x_curr;
        x_rem(i) = 0;
        res = b - A*x_rem;
        
        %update ith coordinate
        %temp = norm(A(:, i), 2)^2;
        den = col_norm_sqr(i);
        x_curr(i) = prox_op_l1(A(:, i)'*res/den, prox_thres(i));
    end
    
    % convergence check
    delta_vec(nIter) = norm(x_curr - x_prev, 2)^2;
    delta = delta_vec(nIter);
    val(nIter) = eval_lasso(A, b, lambda, x_curr); 
end

% pack output variables in one struct
cd_lasso.x_opt = x_curr; 
cd_lasso.nIter = nIter;
cd_lasso.delta_vec = delta_vec;
cd_lasso.val = val; 
end
