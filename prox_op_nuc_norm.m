function Y = prox_op_nuc_norm(X, lambda)
    [U, S, V] = svd(X);
    S_Y = prox_op_l1(diag(S), lambda);
    sz_V = size(V');
    sz_Y = size(S_Y);
    missing_dim = sz_V(1) - sz_Y(1);
    Y = U*[diag(S_Y) zeros(sz_Y(1), missing_dim)]*V';
end