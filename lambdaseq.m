function x = lambdaseq(lambda, lambda1, nfeatures, type)
%LAMBDASEQ - Generate lambda sequence for GrOWL regularization
% The hyperparameters `lambda` and `lambda1` define, essentially, the
% minimum value and range of sequence, respectively. Note, that this
% function defines the exponential case differently than in the `WISC_MVPA`
% toolbox, where it is defined as:
%
%     lambda*sqrt(2*log((nfeatures*ones(1, nfeatures)) ./ (1:nfeatures)))
%
% The `WISC_MVPA` definition includes only lambda, and uses it to scale the
% range of the sequence. The minimum value in the sequence is always zero
% under this definition; it does not allow adjusting the minimum value, and
% it uses `lambda` in a way that is inconsistent with the other
% definitions. It also allows for a maximum value that increases with the
% number of features, which is different from the other sequence types. The
% definition below improves consistency for the definition of `lambda` and
% `lambda1` and also constrains the max value to be `lambda + lambda1`.
%
% All cases generate monotonically decreasing sequences. `lambda`
% and `lambda1` must be positive scalars.
%
% Syntax: x = lambdaseq(lambda, lambda1, nfeatures, type)
%
% Inputs:
%   lambda: minimum value in the sequence
%   lambda1: the range of the sequence. `lambda + lambda1` yields the
%            maximum value.
%   nfeatures: The length of the sequence
%   type: One of 'linear', 'exponential', or 'inf'. The linear sequence 
%
% Outputs: A numeric vector of length `nfeatures`. Values monotonically
%     decrease.
%
% Examples:
%   x = lambdaseq(1,6,10,'linear');
%   x = lambdaseq(1,6,10,'exponential');
%   x = lambdaseq(1,6,10,'inf');
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none

% Created 19-April-2023 by Chris Cox
% Derived from logic in WISC_MVPA/src/learn_encoding.m
    arguments
        lambda (1,1) double {mustBeNonnegative}
        lambda1 (1,1) double {mustBeNonnegative}
        nfeatures (1,1) double {mustBeNonnegative, mustBeInteger}
        type (1,1) string {mustBeTextScalar}
    end

    switch lower(type)
        case 'linear'
            x = lambda1 * (nfeatures:-1:1) / nfeatures + lambda;
        case 'exponential'
            y = sqrt(2 * log((nfeatures * ones(1, nfeatures)) ./ (1:nfeatures)));
            x = (lambda1 * (y / max(y))) + lambda;
        case 'inf'
            x = [lambda + lambda1, repmat(lambda, 1, nfeatures - 1)];
        otherwise
            error('Unknown type `%s`', type);
    end
end

