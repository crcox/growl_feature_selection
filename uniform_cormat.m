function R = uniform_cormat(n, r)
    R = ones(n) * r;
    R(1:n+1:end) = 1;
end