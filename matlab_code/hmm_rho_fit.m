function hmm_rho_fit(experiment, postfix)

    if nargin < 2
        experiment = 'sealion';   % 'sealion', 'turtle', 'sim'
        postfix = '';
    end

    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    % Define the folder where the .mat files are stored.
    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    if ~exist(fdir,'dir')
        error("Experiment folder not found: %s", fdir);
    end

    %% Load data
    if strcmpi(experiment, 'sim')
        fname = sprintf("data_sim_binary_hmm_rho%s.mat", postfix);
        f = load(fullfile(fdir, fname));
        data = f.sim_data;
        fprintf("Using simulated data file to fit: %s\n", fname);
    else
        [data] = get_data(experiment);
    end
    init_fname = fullfile(fdir, sprintf('hmm_rho_blockwise_fit_%s.mat', experiment));
    init_f = load(init_fname);

    N = numel(data);
    blocks = size(data{1}.outcome, 2);

    fname = fullfile(fdir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));
    fprintf("CBM output file: %s\n", fname);
    % SUBJECT-LEVEL MODEL FITTING
    % Parameters per subject: 4 volatility + 4 stochasticity + 1 rho = 9 total
    if ~exist(fname, 'file')

        number_of_parameters = 9;
        prior_variance = 100;

        cbm = cell(1, N);
        workerIds = cell(1, N);

        for n = 1:N

            prior = struct( ...
                'mean', zeros(number_of_parameters,1), ...
                'variance', prior_variance);

            config.verbose = 0;
            config.numinit = 3;

            init_params= init_f.cbm{n}.output.parameters;
            config.inits = [init_params(1:8) mean(init_params(9:12))];

            dat = struct( ...
                'outcome', data{n}.outcome, ...
                'choice',  data{n}.choice );

            cbm{n} = cbm_lap({dat}, @model2fit, prior, '', config);

            if strcmpi(experiment,'sim')
                workerIds{n} = n;
            else
                workerIds{n} = data{n}.workerId;
            end

            fprintf('Processing subject %d/%d\n', n, N);
        end

        save(fname, "cbm", "workerIds");
        fprintf("HMM (shared rho) fit saved.\n");

    else
        fprintf("HMM fit already exists.\n");
    end

    % EXTRACT TRANSFORMED PARAMETERS
    f = load(fname);
    cbm = f.cbm;

    if strcmpi(experiment, 'sim')
        filename = 'hmm_rho_params_sim';
    else
        filename = 'hmm_rho_params';
    end

    fname2 = fullfile(fdir, sprintf('%s%s.mat', filename, postfix));
    fprintf("Fitted params file: %s\n", fname2);

    if ~exist(fname2, 'file')

        lr = nan(N, blocks);
        block_effect = nan(N, blocks);
        parameters = nan(N, 9);

        for n = 1:N

            dat = struct( ...
                'outcome', data{n}.outcome, ...
                'choice',  data{n}.choice );

            [~, parameters(n,:), lr(n,:), block_effect(n,:)] = ...
                model2fit(cbm{n}.output.parameters, dat);

        end

        save(fname2, "lr", "block_effect", "parameters");
        fprintf("HMM parameters saved.\n");

    else
        fprintf("HMM parameters already exist.\n");
    end

end

function [loglikelihood, transformed_parameters, lr, block_effect] = model2fit(x, data)
    % Data
    outcome = data.outcome;
    choices = data.choice;

    outcome(isnan(choices)) = NaN;

    %Parameter transformation
    half_sigmoid = @(x) 0.5 ./ (1 + exp(-x));

    % 4 vol + 4 sto + 1 rho
    vol = half_sigmoid(x(1:4));
    sto = half_sigmoid(x(5:8));
    rho = x(9);     % ONE rho per subject

    ndim = size(outcome, 2);

    predictions = nan(size(outcome));
    lr = nan(1, ndim);
    block_effect = nan(1, ndim);

    % HMM predictions
    for i = 1:ndim
        [predictions(:, i), b] = hmm(outcome(:, i), vol(i), sto(i));
        lr(i) = b(1);
        block_effect(i) = b(2);
    end

    % Response model with shared rho
    resp_params = struct('rho', rho);
    out = response_model(predictions, choices, resp_params);

    loglikelihood = out.loglik;

    % Return transformed parameters
    transformed_parameters = [vol, sto, rho];
end

