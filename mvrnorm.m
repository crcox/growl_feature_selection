function x = mvrnorm(n, cormat)
%MVRNORM Generate gaussian random variables with correlation structure
%
% Inputs:
%    n - number of examples to generate
%    cormat - a correlation matrix (symmetric and positive definite)
%
% Outputs:
%    x - A matrix with `n` rows and `size(cormat, 1)` correlated columns.
%
% Examples:
%   R = [1, .5; .5, 1];
%   x = mvrnorm(1000, R);
%   corr(x(:,1), x(:,2));
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none

% Created 19-April-2023 by Chris Cox
% Solution taken from https://www.mathworks.com/matlabcentral/answers/101802-how-can-i-generate-two-correlated-random-vectors-with-values-drawn-from-a-normal-distribution#answer_111150
    arguments
        n (1,1) double {mustBeInteger}
        cormat (:,:) double {mustBeNumeric, mustBeSymmetric, mustHaveOnesOnDiagonal}
    end
    x = zscore(randn(n, size(cormat, 1)) * chol(cormat));
end

function mustBeSymmetric(x)
    if (~issymmetric(x))
        eidType = 'mustBeSymmetric:notSymmetric';
        msgType = 'A correlation matrix must be symmetric.';
        throwAsCaller(MException(eidType, msgType));
    end
end

function mustHaveOnesOnDiagonal(x)
    if (~all(diag(x) == 1))
        eidType = 'mustHaveOnesOnDiagonal:notHaveOnesOnDiagonal';
        msgType = 'All diagonal elements of a correlation matrix must be 1.';
        throwAsCaller(MException(eidType, msgType));
    end
end