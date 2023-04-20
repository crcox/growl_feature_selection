nitems = 100;
nfeatures = 100;
ndim = 1;

ncorsig = 5;
nuncorsig = 5;
nnoise = nfeatures - ncorsig - nuncorsig;

X = [mvrnorm(nitems, uniform_cormat(ncorsig, .5)), randn(nitems, nuncorsig), randn(nitems, nnoise)];
beta = [ones(ncorsig + nuncorsig, 1); zeros(nnoise, 1)];
noise = randn(nitems, ndim);

Y = (X * beta) + noise;

training_set = ones(nitems, 1);

lamseq = lambdaseq(10, 20, nfeatures, 'linear');

mod = Adlas(lamseq(:), training_set, bias=true);
mod = mod.train(X, Y);
b = mod.getWeights();
bar(b)

%% Permutation test
nperm = 1000;
Yperm = zeros(nitems, nperm);
modperm = repmat(Adlas(lamseq(:), training_set, bias=true), 1, nperm);
for i = 1:1000
    Yperm(:, i) = Y(randperm(nitems), :);
    modperm(i) = modperm(i).train(X, Yperm(:, i));
end
bperm = [modperm.X]; % Weights are stored in X internally
count = sum(bperm ~=0, 2);
bar(count)
