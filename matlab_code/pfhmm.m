function [val, vars] = pfhmm(parameters, observations, config)
% PFHMM Particle Filter HMM (Hidden Markov Model) wrapper.
%
%   [val, vars] = pfhmm(parameters, observations, config)
%
%   This function applies a particle filter model to one or more observation
%   sequences (each column in 'observations') using specified parameters and
%   configuration settings. It collects the output value (val) and additional
%   variables (vars) from the particle filter process.
%
%   Inputs:
%       parameters   - Structure containing model parameters (e.g., s0, v0,
%                      sigma_sto, sigma_vol, etc.).
%       observations - A matrix of observed outcomes, where each column is a
%                      separate sequence.
%       config       - (optional) Structure with configuration settings for the
%                      particle filter (e.g., resampling strategy, number of particles).
%
%   Outputs:
%       val  - A matrix of filtered estimates (one row per time point, one column
%              per observation sequence).
%       vars - A structure containing additional variables from the filtering
%              process (e.g., volatility, stochasticity, learning rate, etc.).

    % If the configuration (config) is not provided, set a default resampling strategy.
    if nargin < 3
        config.resampling_strategy = 'systematic';
    end

    % Use an input parser to handle configuration options with defaults.
    p = inputParser;
    p.addParameter('resampling_strategy', 'systematic');
    p.addParameter('resample_percentage', .5);  % Fraction of particles below which to resample
    p.addParameter('num_particles', 10000);       % Number of particles in the filter
    p.parse(config);
    config = p.Results;

    % Preallocate 'val' to store the filtered state estimates for each observation.
    val = nan(size(observations)); 

    % Process each column (each observation sequence) separately.
    for i = 1:size(observations, 2)
        % Call the core particle filter process (pf_process) for this observation.
        [val(:, i), vars_i] = pf_process(parameters, observations(:, i), config);
        
        % Iterate through all fields returned by pf_process and collect them in 'vars'.
        snames = fieldnames(vars_i);
        for j = 1:length(snames)
            if i == 1
                % For the first observation, initialize each field in vars.
                vars.(snames{j}) = vars_i.(snames{j});
            else
                % For subsequent observations, append the data to each field.
                vars.(snames{j})(:, i) = vars_i.(snames{j});
            end
        end
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PF_PROCESS: Core particle filter processing for one observation sequence
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [val, vars] = pf_process(parameters, outcomes, config)
% PF_PROCESS Processes a single observation sequence using a particle filter.
%
%   [val, vars] = pf_process(parameters, outcomes, config)
%
%   Inputs:
%       parameters - Model parameters (e.g., s0, v0, sigma_sto, sigma_vol).
%       outcomes   - A column vector of observed outcomes for T time points.
%       config     - Configuration options (number of particles, resampling strategy).
%
%   Outputs:
%       val  - Filtered estimate of the hidden state at each time point.
%       vars - Structure containing:
%                vol         - Filtered volatility estimates.
%                sto         - Filtered stochasticity estimates.
%                val         - The filtered state (same as output 'val').
%                y_std, y_mean - Mean and standard deviation of the predictive likelihood.
%                learning_rate - Estimated learning rate (from linear regression).

    % Extract the number of particles and initialize uniform weights.
    num_particles = config.num_particles;
    weights = ones(num_particles, 1) / num_particles;
    
    % Number of time steps equals the length of the outcomes vector.
    T = length(outcomes);
    
    % Initialize particles for hidden state 'r' (belief about the state).
    r = 0.5 * ones(num_particles, 1);
    % Initialize volatility ('s') and stochasticity ('v') using starting values.
    s = parameters.s0 * ones(num_particles, 1);
    v = parameters.v0 * ones(num_particles, 1);

    % Preallocate storage for filtered values and other tracking variables.
    val    = nan(T, 1);   % Estimated state value at each time step.
    vol    = nan(T, 1);   % Mean volatility at each time step.
    sto    = nan(T, 1);   % Mean stochasticity at each time step.
    y_std  = nan(T, 1);   % Std deviation of the predictive likelihood.
    y_mean = nan(T, 1);   % Mean of the predictive likelihood.

    % Loop over every time point in the outcome sequence.
    for t = 1:T
        % Compute the current estimate as the weighted mean over particles.
        val(t) = sum(r .* weights);

        % Transition step: Update volatility and stochasticity particles.
        [v, s] = transition_func(v, s, parameters);

        % Compute and store weighted means for volatility and stochasticity.
        vol(t) = sum(v .* weights);
        sto(t) = sum(s .* weights);

        % Diffuse the current belief state using the diffusion function.
        q = hmm_diffuse(r, v);

        % Calculate the likelihood for the current outcome.
        % The Bernoulli predictive function computes the likelihood of the observed
        % outcome given the prediction q and stochasticity s.
        y = bernoulli_predictive(outcomes(t), q, s);

        % Store mean and standard deviation of the predictive likelihood.
        y_mean(t) = mean(y);
        y_std(t) = std(y);
        
        % Resample the particles based on the likelihood weights.
        [idx, weights, ~] = resample(y, weights, config);
        % Reorder the volatility, stochasticity, and predictions after resampling.
        s = s(idx);
        v = v(idx);
        q = q(idx);

        % Update the state estimate using the HMM update rule.
        r = hmm_update(outcomes(t), s, q);                      
    end

    % After processing all time points, estimate the learning rate.
    % A simple linear regression is run on the prediction error between successive time steps.
    X = outcomes(1:end-1,:) - val(1:end-1,:);
    y_reg = val(2:end, :) - val(1:end-1, :);
    % The GLM estimates an intercept and slope; here we take the slope (learning rate).
    learning_rate = glmfit(X, y_reg, 'normal'); % Linear regression including an intercept
    learning_rate = learning_rate(2);            % Extract the learning rate coefficient
    
    % Pack all computed variables into the output structure.
    vars = struct('vol', vol, 'sto', sto, 'val', val, 'y_std', y_std, ...
                  'y_mean', y_mean, 'learning_rate', learning_rate);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TRANSITION FUNCTION: Update volatility and stochasticity particles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [v_new, s_new] = transition_func(v, s, params)
% TRANSITION_FUNC Applies a stochastic transition to volatility and stochasticity.
%
%   [v_new, s_new] = transition_func(v, s, params)
%
%   Inputs:
%       v     - Current volatility particles.
%       s     - Current stochasticity particles.
%       params- Parameters structure containing noise parameters (sigma_sto, sigma_vol).
%
%   Outputs:
%       v_new - Updated volatility particles.
%       s_new - Updated stochasticity particles.
%
%   A beta distribution is used here to generate noise around the current estimates.

    % Extract noise scale parameters.
    sigma_s = params.sigma_sto;
    sigma_v = params.sigma_vol;

    % Define an inline function to generate beta-distributed noise.
    % The parameters of the beta distribution are scaled based on mu (the current value) and sigma.
    gen_noise = @(mu, sigma)(betarnd(mu ./ sigma, (1 - mu) ./ sigma));
    
    % Update particles for stochasticity and volatility.
    s_new = 0.5 * gen_noise(2 * s, sigma_s + eps);
    v_new = 0.5 * gen_noise(2 * v, sigma_v + eps);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LIKELIHOOD FUNCTION: Bernoulli predictive likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function y = bernoulli_predictive(o, q, s)
% BERNOULLI_PREDICTIVE Computes the likelihood of the observed outcome under
% a Bernoulli model given prediction and noise parameters.
%
%   y = bernoulli_predictive(o, q, s)
%
%   Inputs:
%       o - The observed outcome (0 or 1).
%       q - The predicted probability of outcome 1.
%       s - Stochasticity affecting the outcome.
%
%   Outputs:
%       y - The likelihood value computed using a Bernoulli model.
%
%   The effective probability p blends the prediction (q) and its complement based
%   on the noise (s).

    % Calculate the effective probability of observing 1.
    p = (1 - s) .* q + s .* (1 - q);
    
    % Compute the Bernoulli likelihood:
    %   If o == 1, likelihood is p; if o == 0, likelihood is (1-p).
    y = p .^ o .* (1 - p) .^ (1 - o);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RESAMPLE FUNCTION: Particle resampling based on likelihood weights
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [idx, weights, resampled] = resample(likelihood, weights, config)
% RESAMPLE Resamples the particles based on their likelihood weights.
%
%   [idx, weights, resampled] = resample(likelihood, weights, config)
%
%   Inputs:
%       likelihood - Likelihood values for each particle.
%       weights    - Current weights for each particle.
%       config     - Configuration structure with fields:
%                     resample_percentage and resampling_strategy.
%
%   Outputs:
%       idx        - Indices of resampled particles.
%       weights    - Updated (resampled) weights.
%       resampled  - Indicator (1 if resampling occurred, 0 otherwise).

    % Total number of particles.
    NumParticles = length(likelihood);

    % Multiply current weights by likelihood; add eps for numerical stability.
    weights = weights .* (likelihood + eps);
    weights = weights / sum(weights);

    % Compute effective number of particles.
    Neff = 1 / sum(weights .^ 2);
    
    % Determine the resampling threshold.
    resample_percentage = config.resample_percentage;
    Nt = resample_percentage * NumParticles;
    idx = 1:NumParticles;
    resampled = 0;

    % Check for numerical issues.
    if any(weights < 0)
        disp('There are elements in the array that are <= 0.');
    elseif any(isnan(weights))
        disp('There are NA.')
    end 

    % If effective particles fall below the threshold, resample.
    if Neff < Nt
        N  = length(weights);
        switch config.resampling_strategy
            case 'systematic'
                % Compute cumulative sum of weights with protection from round-off.
                edges = min([0; cumsum(weights)], 1);
                % Generate uniformly spaced numbers with a random start.
                u = (0:1/N:1 - 1/N) + rand * (1/N);
                % Discretize the u values to obtain resampled indices.
                idx = discretize(u, edges);
            case 'multinomial'
                % Draw random numbers and use cumulative weights to determine indices.
                u = rand(N, 1);
                wc = cumsum(weights');
                wc = wc(:) / wc(N);
                [~, ind1] = sort([u; wc]);
                ind2 = find(ind1 <= N);
                idx = ind2' - (0:N-1);
            otherwise
                error('Resampling strategy is unknown');
        end
        
        % Reset weights to be uniform after resampling.
        weights = ones(size(weights)) / N;
        resampled = 1;
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HMM DIFFUSE FUNCTION: Diffusing the state probability
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function q = hmm_diffuse(r, v)
% HMM_DIFFUSE Diffuses the current state probability using the volatility.
%
%   q = hmm_diffuse(r, v)
%
%   Inputs:
%       r - Current state estimate for each particle.
%       v - Volatility (noise) affecting the state.
%
%   Outputs:
%       q - Diffused state probability computed as a weighted combination of r.
%
%   The function computes a new probability that blends r with its complement
%   using v as the weight.

    q = r .* (1 - v) + (1 - r) .* v;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% HMM UPDATE FUNCTION: Update particle filter based on observation likelihood
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function r_new = hmm_update(o, s, q)
% HMM_UPDATE Updates the state probability based on the new observation.
%
%   r_new = hmm_update(o, s, q)
%
%   Inputs:
%       o - The observed outcome (0 or 1).
%       s - Stochasticity vector (noise affecting the outcome).
%       q - Diffused state probability vector.
%
%   Outputs:
%       r_new - Updated state probability computed using Bayesian updating.
%
%   This function computes a likelihood-based update:
%       - It first computes a likelihood (ell) blending the outcome with noise.
%       - Then updates the state probability using normalization.

    % Compute the likelihood factor (ell) from the outcome and noise.
    ell = o .* (1 - s) + (1 - o) .* s;    
    
    % Update the state probability r_new using a Bayesian-like formula.
    % A small eps is added to avoid division by zero.
    r_new = ell .* q ./ (ell .* q + (1 - ell) .* (1 - q) + eps);
end

