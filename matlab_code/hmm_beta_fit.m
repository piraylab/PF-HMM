function hmm_beta_fit(experiment)
% HMM_FIT Fits HMM model to experimental data.
%
%   hmm_beta_fit(experiment)
%
%   This function loads experimental data for the specified experiment,
%   fits a HMM with CBM package per subject (and block) using a
%   Laplace approximation via cbm_lap, and then computes HMM parameters 
%   (learning rate and block effect) via model2fit.
%
%   Inputs:
%       experiment - A string specifying which experiment data to load.
%                    Defaults to 'sealion' if not provided.
%
%   The function saves two files:
%       1. 'cbm_fit.mat'  - Contains fitting results per subject.
%       2. 'hmm_params.mat' - Contains learning rate, block effect, parameters,
%                            and worker IDs.
%

    % Set default experiment if none provided.
    if nargin < 1, experiment = 'sealion'; end

    %% Set up paths and directories
    % Determine the current folder (location of this file) and set the working directory
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    
    % Add folder 'cbm' to the MATLAB path (assumes helper functions are in this folder)
    addpath('cbm');
    addpath('tools');
    
    % Define the folder where the .mat data files are stored
    fdir = fullfile('..', 'mat_data');

    %% Load experimental data using the get_data function
    [data, ~] = get_data(experiment);
    N = length(data);  % number of subjects (data entries)

    %% Fit the CBM model per subject and per block
    % Define the filename to save CBM fitting results
    fname = fullfile(fdir, sprintf('%s.mat', mfilename)); 
    if ~exist(fname, 'file')
        % Set the number of parameters to be estimated: v (volatility), s (noise), beta (inverse temperature)
        number_of_parameters = 3; 
        v = 6.25; % Variance (used in the prior for cbm_lap)
        
        % Initialize a cell array to store CBM fit results for each subject.
        cbm = cell(1, N);
        
        % Loop over each subject
        for n = 1:N
            % Preallocate arrays for the parameters and log evidence per block (4 blocks per subject)
            parameters = nan(4, number_of_parameters);
            log_evidence = nan(4, 1);
            
            % Loop over each block for the given subject
            for i = 1:4
                % Set up the prior structure with zero mean and fixed variance for each parameter
                prior = struct('mean', zeros(number_of_parameters,1), 'variance', v);
                % Configure number of initializations and verbosity for the cbm_lap procedure
                config.verbose = 0;
                
                % Prepare a data structure for the current block containing outcome and choice vectors.
                dat = struct('outcome', data{n}.outcome(:, i), 'choice', data{n}.choice(:, i));
                
                % Perform the Laplace approximation model fitting using the helper function model2fit.
                % Note: The function cbm_lap is assumed to be available in the added path.
                [cbm_blk] = cbm_lap({dat}, @model2fit, prior, '', config);
                
                % Store the fitted parameters and log evidence for the current block.
                parameters(i, :) = cbm_blk.output.parameters;
                log_evidence(i) = cbm_blk.output.log_evidence;
            end
            % Reshape the parameters from 4 blocks into a single row (12 values per subject)
            cbm{n}.output.parameters = reshape(parameters, 1, 4*number_of_parameters);
            % Sum the log evidence over all blocks
            cbm{n}.output.log_evidence = sum(log_evidence);        
            
        end
        % Save the CBM fitting results to a .mat file
        save(fname, "cbm");
    end
    
    % Load the CBM fitting results from file
    f = load(fname);
    cbm = f.cbm; 

    %% Calculate HMM (learning rate) parameters using model2fit
    % Define the filename for the HMM parameters and associated data.
    fname = fullfile(fdir, sprintf('hmm_beta_params.mat')); 
    if ~exist(fname, 'file')
        % Preallocate matrices to hold learning rate (lr), block effect, and parameters for each subject
        lr = nan(N, 4);
        block_effect = nan(N, 4);
        parameters = nan(N, 12);
        
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
    end

    %% Call regression function to further analyze the learning rate and block effect
    f = load(fname);
    lr = f.lr;
    block_effect = f.block_effect;
    % compute_effect is assumed to be another function that processes these parameters.
    st = compute_effect(lr, block_effect);
    
    T = array2table(st.table.data, 'VariableNames', st.table.columns, ...
        'RowNames', st.table.rows);
    fprintf('\n=== Table: Fitted HMM-Beta Regression Effect Table ===\n');
    disp(T);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function: model2fit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [loglikelihood, transformed_parameters, lr, block_effect] = model2fit(x, data)
% MODEL2FIT Transforms parameters and computes the log likelihood using a Hidden Markov Model.
%
%   [loglikelihood, transformed_parameters, lr, block_effect] = model2fit(x, data)
%
%   Inputs:
%       x    - Parameter vector; can be of length 3, 9, or 12. Depending on its
%              length, parameters represent volatility (vol), noise (sto), and beta.
%       data - Structure containing:
%                .outcome : Outcome matrix (trials x dimensions)
%                .choice  : Choice matrix (trials x dimensions). Missing or invalid
%                          choices are noted as NaN.
%
%   Outputs:
%       loglikelihood         - The summed log likelihood over trials given the model.
%       transformed_parameters- The parameters after transformation:
%                               vol: scaled between 0 and 0.5,
%                               sto: scaled between 0 and 0.5,
%                               beta: exponentiated so that beta > 0.
%       lr                    - Estimated learning rate (first coefficient from GLM) for each dimension.
%       block_effect          - Secondary coefficient (e.g., block-level effect) from GLM for each dimension.
%

    %% Parse and clean data
    % Extract outcome and choice matrices from data
    outcome = data.outcome;
    choices = data.choice;
    
    % Ensure that if choices are missing (NaN), the corresponding outcome is set to NaN.
    outcome(isnan(choices)) = NaN;
    
    %% Transform parameters using half sigmoid and exponential functions
    % Define a half-sigmoid function that maps input x to the range (0, 0.5)
    half_sigmoid = @(x) 0.5 ./ (1 + exp(-x));
    
    % Depending on the length of parameter vector x, transform parameters accordingly.
    if length(x) == 12
        % For 12 parameters: first 4 values for vol, next 4 for sto, and last 4 for beta
        vol = half_sigmoid(x(1:4)); % Scaled to (0, 0.5)
        sto = half_sigmoid(x(5:8)); % Scaled to (0, 0.5)
        beta = exp(x(9:12));        % Exponentiated so that beta > 0
    elseif length(x) == 3
        % For 3 parameters: one value each for vol, sto, and beta (shared across dimensions)
        vol = half_sigmoid(x(1));   % Scaled to (0, 0.5)
        sto = half_sigmoid(x(2));   % Scaled to (0, 0.5)
        beta = exp(x(3));           % Exponentiated so that beta > 0
    end
    % Combine transformed parameters into one vector for output.
    transformed_parameters = [vol, sto, beta];
    
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
    % Calculate the probability of choices by raising the predictions to the power of beta.
    % (Note that beta adjusts the "sharpness" of the choice probabilities.)
    prob_choice1 = predictions .^ beta;
    prob_choice2 = (1-predictions) .^ beta;
    prob_choice1 = prob_choice1./(prob_choice1 + prob_choice2);
    
    % Compute log likelihood for each trial and dimension.
    % Add a small number (eps) to avoid taking the log of zero.
    loglikelihood = log(prob_choice1 + eps) .* choices + log(1 - prob_choice1 + eps) .* (1 - choices);

    % Remove contribution of trials with missed or invalid choices (NaN)
    loglikelihood(isnan(choices)) = 0;
    
    % Sum the log likelihoods over all trials and dimensions.
    loglikelihood = sum(sum(loglikelihood));
    
end



