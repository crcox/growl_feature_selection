nitems = 100;
nfeatures = 10;
ndim = 4;
beta = randn(nfeatures, ndim);
X = randn(nitems, nfeatures);
noise = randn(nitems, ndim);
Y = (X * beta) + noise;

cvblocks = cvpartition(nitems, "KFold", 10);
training_set = training(cvblocks, 1);
%training_set = ones(nitems, 1);

HAS_BIAS = 1;
opts = struct( ...
    'lambda', 2, ...
    'lambda1', 10 ...
);

lamseq = lambdaseq(2, 10, nfeatures, 'linear');

for repetition = 1:10
    mod = Adlas(size(X), size(Y), lamseq(:), training_set, HAS_BIAS, opts);
    mod = mod.train(X, Y, opts);
    mod = mod.test(X, Y);
    disp([mod.trainingError, mod.testError])
end
