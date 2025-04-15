function pfhmm_sim
% PFHMM_SIM Simulates hidden state estimation using a particle filter HMM.
%
%   This function runs a simulation across multiple iterations (num_sim) and
%   a set number of trials (num_trials). For each simulation, it generates an
%   outcome sequence using a hidden state process (via the gen_outcome helper
%   function). It then applies different parameter settings to simulate
%   learning rates and volatility/stochasticity estimates under three conditions:
%       - "Healthy": both volatility and stochasticity noise are present.
%       - "VMAL": volatility noise is minimized (set to near zero).
%       - "SMAL": stochasticity noise is minimized (set to near zero).
%   It also computes the true HMM parameters using hmm (for simulation plots)
%   and saves all results into separate MAT-files.

    %% Define true parameters and simulation settings
    % Create a metadata structure with true values for stochasticity and volatility
    metadata = struct('true_sto', [0.1250 0.1250 0.2500 0.2500], ... 
                      'true_vol', [0.0250 0.1000 0.0250 0.1000]);
    
    % Set number of trials per simulation and total number of simulations
    num_trials = 100;
    num_sim = 1000;

    %% Preallocate variables to store simulation outputs
    % Store true parameters estimated from the HMM (8 parameters per simulation)
    b_true = zeros(num_sim, 8);

    % Preallocate matrices for per-block learning rates under three conditions:
    % lr_healthy: both volatility and stochasticity noise are present.
    % lr_vmal: minimal volatility noise (vmal = volatility minimal).
    % lr_smal: minimal stochasticity noise (smal = stochasticity minimal).
    lr_healthy = zeros(num_sim, 4);
    lr_vmal = zeros(num_sim, 4);
    lr_smal = zeros(num_sim, 4);

    % Preallocate cell arrays for the full volatility and stochasticity time series
    % along with matrices to store the mean values per block.
    vol_healthy = cell(num_sim, 1);
    sto_healthy = cell(num_sim, 1);
    m_vol_healthy = zeros(num_sim, 4);
    m_sto_healthy = zeros(num_sim, 4);

    vol_vmal = cell(num_sim, 1);
    sto_vmal = cell(num_sim, 1);
    m_vol_vmal = zeros(num_sim, 4);
    m_sto_vmal = zeros(num_sim, 4);

    vol_smal = cell(num_sim, 1);
    sto_smal = cell(num_sim, 1);
    m_vol_smal = zeros(num_sim, 4);
    m_sto_smal = zeros(num_sim, 4);

    %% Loop over simulations
    for n = 1:num_sim
        % Generate a simulated outcome sequence based on the true metadata.
        % The random seed (n) ensures each simulation is reproducible.
        [o] = gen_outcome(metadata, num_trials, n);

        % Extract the true volatility and stochasticity from metadata.
        v = metadata.true_vol;
        s = metadata.true_sto;    
        % Compute the true HMM parameters using a core function and store them.
        % 'hmm' is assumed to use these true parameters to generate b_true.
        [~, b_true(n,:)] = hmm(o, v, s); 

        %% Simulation under "Healthy" condition
        % Set random seed to ensure reproducible noise for this simulation.
        rng(n);
        % Specify parameter noise for healthy condition: both sigma_sto and sigma_vol are nonzero.
        params.sigma_sto = 0.02;
        params.sigma_vol = 0.02;
        params.s0 = mean(metadata.true_sto);
        params.v0 = mean(metadata.true_vol);
        % Apply the particle filter HMM (pfhmm) to the generated outcome sequence.
        [~, vars] = pfhmm(params, o);
        % Record the learning rate and full time series estimates.
        lr_healthy(n, :) = vars.learning_rate;
        vol_healthy{n} = vars.vol;
        sto_healthy{n} = vars.sto;
        m_vol_healthy(n, :) = mean(vars.vol);  
        m_sto_healthy(n, :) = mean(vars.sto);  

        %% Simulation under "VMAL" condition (minimal volatility noise)
        rng(n);
        % Set volatility noise very low (using eps) while keeping stochasticity noise unchanged.
        params.sigma_sto = 0.02;
        params.sigma_vol = eps;  % eps nearly zero, effectively minimizing volatility noise.
        params.s0 = mean(metadata.true_sto);
        params.v0 = mean(metadata.true_vol);
        [~, vars] = pfhmm(params, o);
        % Store estimates for the VMAL condition.
        lr_vmal(n, :) = vars.learning_rate;
        vol_vmal{n} = vars.vol;
        sto_vmal{n} = vars.sto;
        m_vol_vmal(n, :) = mean(vars.vol);  
        m_sto_vmal(n, :) = mean(vars.sto);   

        %% Simulation under "SMAL" condition (minimal stochasticity noise)
        rng(n);
        % Set stochasticity noise very low (using eps) while keeping volatility noise nonzero.
        params.sigma_sto = eps;
        params.sigma_vol = 0.02;
        params.s0 = mean(metadata.true_sto);
        params.v0 = mean(metadata.true_vol);
        [~, vars] = pfhmm(params, o);
        % Record the estimates for the SMAL condition.
        lr_smal(n, :) = vars.learning_rate;  
        vol_smal{n} = vars.vol;
        sto_smal{n} = vars.sto;
        m_vol_smal(n, :) = mean(vars.vol);
        m_sto_smal(n, :) = mean(vars.sto);      

        % Print progress to the console to monitor simulation progress.
        fprintf('\rProgress %d/%d', n, num_sim);
    end

    %% Save simulation outputs to .mat files
    % Change directory back to the current file's folder.
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    % Define the directory in which the simulation results will be saved.
    fdir = fullfile('..', 'mat_data');
    % Save each condition's results into separate MAT-files.
    save(fullfile(fdir, 'pfhmm_healthy.mat'), 'lr_healthy', 'vol_healthy', 'sto_healthy', 'm_vol_healthy', 'm_sto_healthy');
    save(fullfile(fdir, 'pfhmm_vmal.mat'), 'lr_vmal', 'vol_vmal', 'sto_vmal', 'm_vol_vmal', 'm_sto_vmal');
    save(fullfile(fdir, 'pfhmm_smal.mat'), 'lr_smal', 'vol_smal', 'sto_smal', 'm_vol_smal', 'm_sto_smal');
    % Save the true HMM parameter estimates for simulation plotting.
    save(fullfile(fdir, 'hmm_sim.mat'), 'b_true');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function: Generate Outcome Sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [outcome] = gen_outcome(metadata, num_trials, seed)
% GEN_OUTCOME Generates a simulated outcome sequence from a hidden state model.
%
%   [outcome] = gen_outcome(metadata, num_trials, seed)
%
%   Inputs:
%       metadata   - Structure that includes true volatility (true_vol) and 
%                    true stochasticity (true_sto) for each block.
%       num_trials - Number of trials to simulate.
%       seed       - Random seed for reproducibility.
%
%   Outputs:
%       outcome    - A matrix of simulated outcomes based on the underlying state.
%
%   The function simulates a hidden state sequence and then generates outcomes that
%   reflect the state with some noise.
    
    % Set the number of trials.
    N = num_trials;
    % Initialize the random number generator using the provided seed.
    rng(seed);
    
    % Extract true volatility and stochasticity from metadata.
    true_vol = metadata.true_vol;
    true_sto = metadata.true_sto;
    
    % Set up transition and outcome probabilities:
    %   - mu: probability of staying in the same state (1 - true_vol).
    %   - omega: probability that the outcome reflects the hidden state (1 - true_sto).
    mu = 1 - true_vol;
    omega = 1 - true_sto;

    %% Initialize the hidden state sequence
    % Preallocate a matrix z to hold the state for each trial and block.
    z = zeros(N, size(true_vol, 2));  
    % Set the initial state for each block using a binomial draw with p = 0.5.
    z(1, :) = binornd(1, 0.5 * ones(size(true_vol)));
    
    %% Generate the state sequence using transition probabilities
    for t = 2:N
        % For each subsequent trial, generate random binary outcomes based on mu.
        rnd = binornd(1, mu);  
        % Update the state:
        %   - If the previous state is 1 and rnd is 1, the state stays 1.
        %   - Otherwise, the state is switched.
        z(t, :) = z(t - 1, :) .* rnd + (1 - z(t - 1, :)) .* (1 - rnd);
    end
    
    %% Generate outcome sequence based on the state sequence
    % Use omega to simulate the chance that the observed outcome correctly reflects the state.
    rnd = binornd(1, omega .* ones(size(z)));  
    % Compute the outcome:
    %   - When rnd == 1, the outcome is the state.
    %   - When rnd == 0, the outcome is the complement of the state.
    outcome = rnd .* z + (1 - rnd) .* (1 - z);
end

