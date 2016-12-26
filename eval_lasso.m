function output = eval_lasso(A, b, lam, x)
    output = lam*norm(x, 1) + 0.5*norm( b - A*x ,2)^2;
end