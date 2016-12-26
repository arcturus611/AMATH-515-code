function output = nuclear_norm(X)
    [U, S, V] = svd(X);
    output = sum(diag(S));
end