%% Simulation parameters
nitems = 100;
nfeatures = 100;
ndim = 1;
ncorsig = 5;
nuncorsig = 5;
nnoise = nfeatures - ncorsig - nuncorsig;


%% Simulate data
X = [mvrnorm(nitems, uniform_cormat(ncorsig, .5)), randn(nitems, nuncorsig), randn(nitems, nnoise)];
beta = [ones(ncorsig + nuncorsig, 1); zeros(nnoise, 1)];
noise = randn(nitems, ndim);
Y = (X * beta) + noise;


%% Set model paramters
% The help for lambdaseq describes lambda and lambda1
training_set = ones(nitems, 1);
lamseq = lambdaseq(10, 20, nfeatures, 'linear');


%% Fit model to "true" targets 
mod = Adlas(lamseq(:), training_set, bias=true);
mod = mod.train(X, Y);
b = mod.getWeights();


%% Fit models to permuted targets
nperm = 1000;
Yperm = zeros(nitems, nperm);
modperm = repmat(Adlas(lamseq(:), training_set, bias=true), 1, nperm);
for i = 1:1000
    Yperm(:, i) = Y(randperm(nitems), :);
    modperm(i) = modperm(i).train(X, Yperm(:, i));
end
bperm = cell2mat(arrayfun(@(x) x.getWeights(), modperm, 'UniformOutput', false));
count = sum(bperm ~=0, 2);
avgmag = mean(abs(bperm), 2);


%% Plot solutions
color_map = struct( ...
    'correlated_features', [223 145 167], ...
    'uncorrelated_features', [178 198 65], ...
    'noise_features', [134 156 211] ...
);
clr = [
    repmat(color_map.correlated_features, ncorsig, 1);
    repmat(color_map.uncorrelated_features, nuncorsig, 1); 
    repmat(color_map.noise_features, nnoise, 1);
] / 255;


f = figure('Position', [0, 0, 1800, 400]);
subplot(1, 3, 1);
bp = bar(b, 'FaceColor', 'flat'); bp.CData = clr;
title("Model weights fit to true target structure")

subplot(1, 3, 2);
bp = bar(count, 'FaceColor', 'flat'); bp.CData = clr;
title("Non-zero weight counts over permuted target structures")

subplot(1, 3, 3);
bp = bar(avgmag, 'FaceColor', 'flat'); bp.CData = clr;
title("Average weight magnitude over permuted target structures")

exportgraphics(f,'growl_features_permutation.jpg','Resolution',150);