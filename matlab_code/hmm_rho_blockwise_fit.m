function hmm_rho_blockwise_fit(experiment, postfix)
    if nargin < 2
        experiment = 'sim'; % 'sealion', 'turtle', 'sim'
        postfix = ''; 
    end

    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    if ~exist(fdir,'dir'), error("Experiment folder not found: %s", fdir); end
    
    if strcmpi(experiment, 'sim')
        f = load(fullfile(fdir, sprintf("data_sim_binary_hmm_rho%s.mat", postfix)));
        data = f.sim_data;
    else
        [data] = get_data(experiment);
    end
    N = numel(data);
    blocks = size(data{1}.outcome, 2);

    fname = fullfile(fdir, sprintf('%s_%s%s.mat', mfilename, experiment, postfix));
    
    if ~exist(fname, 'file')
        
        % Set the number of parameters to be estimated: v (volatility), s (noise), rho
        number_of_parameters = 3; 
        v = 100; % Variance (used in the prior for cbm_lap)
        
        % Initialize a cell array to store CBM fit results for each subject.
        cbm = cell(1, N);
        workerIds = cell(1, N);
        
        % Loop over each subject
        for n = 1:N
            % Preallocate arrays for the parameters and log evidence per block (4 blocks per subject)
            parameters = nan(blocks, number_of_parameters);
            log_evidence = nan(blocks, 1);
            
            % Loop over each block for the given subject
            for i = 1:blocks
                % Set up the prior structure with zero mean and fixed variance for each parameter
                prior = struct('mean', zeros(number_of_parameters,1), 'variance', v);
                % Configure number of initializations and verbosity for the cbm_lap procedure
                config.verbose = 0;
                config.numinit = 1;
                % Prepare a data structure for the current block containing outcome and choice vectors.
                dat = struct('outcome', data{n}.outcome(:, i), 'choice', data{n}.choice(:, i));
                
                % Perform the Laplace approximation model fitting using the helper function model2fit.
                % Note: The function cbm_lap is assumed to be available in the added path.
                [cbm_blk] = cbm_lap({dat}, @model2fit, prior, '', config); % fourth input can be a filepath to save the whole cbm, cbm.math.Ainvdiag
                
                % Store the fitted parameters and log evidence for the current block.
                parameters(i, :) = cbm_blk.output.parameters;
                log_evidence(i) = cbm_blk.output.log_evidence;
            end
            % Reshape the parameters from 4 blocks into a single row (12 values per subject)
            % cbm{n}.output.parameters = reshape(parameters, 1, 4*number_of_parameters);
            cbm{n}.output.parameters = [ ...
                                        reshape(parameters, 1, blocks*number_of_parameters)];
            % Sum the log evidence over all blocks
            cbm{n}.output.log_evidence = sum(log_evidence);  
            if strcmpi(experiment, 'sim')
                workerIds{n} = n;               % just 1:N for simulated subjects
            else
                workerIds{n} = data{n}.workerId;
            end    
            if mod(n,1)==0
                fprintf('Processing subject %d/%d\n', n, N);
            end
        end
        % Save the CBM fitting results to a .mat file
        save(fname, "cbm", "workerIds");
        fprintf("HMM-fit is saved\n")
    else
        fprintf("HMM-fit already exists\n")
    end
    
    % Load the CBM fitting results from file
    f = load(fname);
    cbm = f.cbm; 
    % workerIds = f.workerIds;

    %% Calculate HMM (learning rate) parameters using model2fit
    % Define the filename for the HMM parameters and associated data.
    if strcmpi(experiment, 'sim')
        filename = 'hmm_rho_blockwise_params_sim';
    else
        filename = 'hmm_rho_blockwise_params';
    end
    fname = fullfile(fdir, sprintf('%s%s.mat', filename, postfix)); 
    if ~exist(fname, 'file')
        % Preallocate matrices to hold learning rate (lr), block effect, and parameters for each subject
        num_params = 3;
        lr = nan(N, 4);
        block_effect = nan(N, 4);
        parameters = nan(N, 4*(num_params-1)+1);
        
        % Loop over each subject and compute HMM parameters using the model2fit function.
        for n = 1:N
            % Structure containing all blocks for the subject
            dat = struct('outcome', data{n}.outcome, 'choice', data{n}.choice);
            
            % model2fit returns log likelihood, transformed parameters, learning rates, and block effects.
            % We store the transformed parameters, learning rate per block, and block effects.
            [~, parameters(n,:), lr(n,:), block_effect(n,:)] = model2fit(cbm{n}.output.parameters, dat);    
            
            % Also store the workerId from the data.
        end
        % Save the HMM parameters to a .mat file
        save(fname, "lr", "block_effect", "parameters");
        fprintf("HMM-params is saved\n")
    else
        fprintf("HMM-params already exists\n")
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function: model2fit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood, transformed_parameters, lr, block_effect] = model2fit(x, data)
    %% Parse and clean data
    outcome = data.outcome;
    choices = data.choice;
    
    % Ensure that if choices are missing (NaN), the corresponding outcome is set to NaN.
    outcome(isnan(choices)) = NaN;
    
    %% Transform parameters using half sigmoid and exponential functions
    % Define a half-sigmoid function that maps input x to the range (0, 0.5)
    half_sigmoid = @(x) 0.5 ./ (1 + exp(-x));
    
    % Depending on the length of parameter vector x, transform parameters accordingly.
    if length(x) == 3
        % For 2 parameters: one value each for vol, sto
        vol = half_sigmoid(x(1));   % Scaled to (0, 0.5)
        sto = half_sigmoid(x(2));   % Scaled to (0, 0.5)
        rho = x(end);
    else
        % For 8 parameters: first 4 values for vol, next 4 for sto
        vol = half_sigmoid(x(1:4)); % Scaled to (0, 0.5)
        sto = half_sigmoid(x(5:8)); % Scaled to (0, 0.5)
        rho = x(9:end);
    end
    
    %% Compute predictions using the HMM function for each dimension
    ndim = size(outcome, 2);        % Number of dimensions (blocks)
    predictions = nan(size(outcome));
    lr = nan(1, ndim);              % Learning rate for each dimension
    block_effect = nan(1, ndim);    % Block effect for each dimension

    % If more than one output is required (i.e. lr and block_effect), compute both.
    if nargout > 1
        for i = 1:ndim
            % Call the HMM update function with the outcome, volatility, and noise
            [predictions(:, i), b] = hmm(outcome(:, i), vol(i), sto(i));
            % b(1) is interpreted as the learning rate, b(2) as the block effect.
            lr(i) = b(1);
            block_effect(i) = b(2);
        end
    else
        for i = 1:ndim
            predictions(:, i) = hmm(outcome(:, i), vol(i), sto(i));
        end
    end
    
    %% Compute the log likelihood for choices

    % --- response model with perseveration ---
    resp_params = struct('rho', mean(rho));
    
    out = response_model(predictions, choices, resp_params);
    loglikelihood = out.loglik;

    % Combine transformed parameters into one vector for output.
    transformed_parameters = [vol, sto, mean(rho)];
    
end



