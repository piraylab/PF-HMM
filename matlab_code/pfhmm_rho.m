function [loglik, val, vars] = pfhmm_rho(parameters, dat, mode, config)

    if strcmp(mode, 'config')
        if strcmp(config.experiment,'binA')
            % -------- bias-based response model --------
            pnames = {'sigma_sto','sigma_vol','bv','be','bi'};
            opt_var = [ ...
                optimizableVariable('sigma_sto',[1e-4 0.5]), ...
                optimizableVariable('sigma_vol',[1e-4 0.5]), ...
                optimizableVariable('bv',[-10 10]), ...
                optimizableVariable('be',[-10 10]), ...
                optimizableVariable('bi',[-10 10]) ...
            ];
            InitialX = array2table([ ...
                0.02  0.02  0   0   0;
                0.10  0.10  0   0   0;
                0.30  0.30  0   0   0], ...
                'VariableNames', pnames);
            rand_func = @(n)[ ...
                        1e-4 + (0.5 - 1e-4) * rand(n,1), ...
                        1e-4 + (0.5 - 1e-4) * rand(n,1), ...
                       -5 + 10 * rand(n,1), ...   % bv
                       -5 + 10 * rand(n,1), ...   % be
                       -5 + 10 * rand(n,1)  ...   % bi
                        ];
        elseif strcmp(config.experiment, 'binB') 
            % -------- beta-based response model --------
            pnames = {'sigma_sto','sigma_vol','temp'};
            opt_var = [ ...
                optimizableVariable('sigma_sto',[1e-4 0.5]), ...
                optimizableVariable('sigma_vol',[1e-4 0.5]), ...
                optimizableVariable('temp',[-10 10]) ...
            ];
    
            InitialX = array2table([ ...
                0.02  0.02  0;
                1e-4  1e-4  0;
                0.1   0.1   0;
                0.5   0.5   0], ...
                'VariableNames', pnames);
            rand_func = @(n)[ ...
                        1e-4 + (0.5 - 1e-4) * rand(n,1), ...   % sigma_sto ∈ (1e-4, 0.5)
                        1e-4 + (0.5 - 1e-4) * rand(n,1), ...   % sigma_vol ∈ (1e-4, 0.5)
                        -5  + 10 * rand(n,1)              ... % rho ∈ (−5, 5)
                        ];
        else
            % -------- perseveration-based response model --------
            pnames = {'sigma_sto','sigma_vol','rho'};
            opt_var = [ ...
                optimizableVariable('sigma_sto',[1e-4 0.5]), ...
                optimizableVariable('sigma_vol',[1e-4 0.5]), ...
                optimizableVariable('rho',[-10, 10]) ... # fitting actual is [-10, 10]
            ];
    
            InitialX = array2table([ ...
                0.02  0.02  0;
                1e-4  1e-4  0;
                0.1   0.1   0;
                0.5   0.5   0], ...
                'VariableNames', pnames);
            rand_func = @(n)[ ...
                        1e-4 + (0.5 - 1e-4) * rand(n,1), ...   % sigma_sto ∈ (1e-4, 0.5)
                        1e-4 + (0.5 - 1e-4) * rand(n,1), ...   % sigma_vol ∈ (1e-4, 0.5)
                        -.5  + 1 * rand(n,1)              ... % rho ∈ (−0.5, 0.5)
                        ];

        end
    
        val = nan;
        vars = struct('opt_var', opt_var, 'InitialX', InitialX, 'parameters_name', {pnames}, 'rand_func', rand_func);
        loglik = NaN;
        % accuracy = NaN;
        return;

    elseif strcmp(mode, 'sim')
        outcomes = dat.outcome;   % ground-truth feedback (0/1)
        [val, vars] = pf_process(parameters, outcomes, mode, config);
        loglik = NaN;
    else    
        outcomes = dat.outcome;   % ground-truth feedback (0/1)
        choices = dat.choice;    % subject prediction (0/1)
        outcomes(isnan(choices)) = NaN;    % mask outcome when choice missing
    
        [val, vars] = pf_process(parameters, outcomes, mode, config);
        % -------- response model selection --------
        if strcmp(config.experiment,'binA')
            % bias-based model (no rho)
            resp_params = struct( ...
                'bv', parameters.bv, ...
                'be', parameters.be, ...
                'bi', parameters.bi ...
            );
            out = response_model_A(val, choices, resp_params);   
        elseif strcmp(config.experiment, 'binB')
            resp_params = struct('temp', parameters.temp);
            trialvalue = dat.trialvalue;
            out = response_model_B(val, trialvalue, choices, resp_params);   
        else
            % perseveration-only model
            resp_params = struct('rho', parameters.rho);
            out = response_model(val, choices, resp_params);
        end
    
        loglik = out.loglik; 

        % % accuracy 
        % stats = size(outcomes,1);
        % t1 = 1:(stats-1);
        % accuracy = (choices == outcomes);
        % accuracy = accuracy(t1,:);
    end
end

% TRANSITION FUNCTION: Update volatility and stochasticity particles
function [v_new, s_new] = transition_func(v, s, params)
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

function [val, vars] = pf_process(parameters, outcomes, mode, config)
    config.resampling_strategy = 'systematic';
    config.resample_percentage = 0.5;
    config.num_particles = 10000;
    
    % p = inputParser;
    % p.addParameter('resampling_strategy', 'systematic');
    % p.addParameter('resample_percentage', .5);  % Fraction of particles below which to resample
    % p.addParameter('num_particles', 10000);       % Number of particles in the filter
    % p.parse(config);
    % config = p.Results;
    % Preallocate 'val' to store the filtered state estimates for each observation.
    val = nan(size(outcomes)); 

    % Process each column (each observation sequence) separately.
    for i = 1:size(outcomes, 2)
        % Call the core particle filter process (pf_process) for this observation.
        [val(:, i), vars_i] = pf_process_block(parameters, outcomes(:, i), config);
        if strcmp(mode, 'fit')        
            vars = [];
        elseif strcmp(mode, 'sim')
            snames = fieldnames(vars_i);
            % Iterate through all fields returned by pf_process and collect them in 'vars'.
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

end

function [val, vars] = pf_process_block(parameters, outcomes, config)
    % Extract the number of particles and initialize uniform weights.
    num_particles = config.num_particles;
    weights = ones(num_particles, 1) / num_particles;
    s0 = config.s0;
    v0 = config.v0;
    
    % Number of time steps equals the length of the outcomes vector.
    T = length(outcomes);
    
    % Initialize particles for hidden state 'r' (belief about the state).
    r = 0.5 * ones(num_particles, 1);
    % Initialize volatility ('s') and stochasticity ('v') using starting values.
    s = s0 * ones(num_particles, 1);
    v = v0 * ones(num_particles, 1);

    % Preallocate storage for filtered values and other tracking variables.
    val    = nan(T, 1);   % Estimated state value at each time step.
    vol    = nan(T, 1);   % Mean volatility at each time step.
    sto    = nan(T, 1);   % Mean stochasticity at each time step.
    y_std  = nan(T, 1);   % Std deviation of the predictive likelihood.
    y_mean = nan(T, 1);   % Mean of the predictive likelihood.

    % Loop over every time point in the outcome sequence.
    for t = 1:T
        % Compute the current estimate as the weighted mean over particles.
        % Predictive mean BEFORE seeing outcome
        val(t) = sum(r .* weights);

        % Transition step: Update volatility and stochasticity particles.
        [v, s] = transition_func(v, s, parameters);
        % Compute and store weighted means for volatility and stochasticity.
        vol(t) = sum(v .* weights);
        sto(t) = sum(s .* weights);

        % Diffuse the current belief state using the diffusion function.
        q = hmm_diffuse(r, v);

        if isnan(outcomes(t))
            % MISSING OUTCOME: prediction-only step
            y_mean(t) = NaN;
            y_std(t)  = NaN;
            % No resampling, no update
            r = q;
            continue
        end

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
    % Compute prediction error and update
    delta = outcomes(1:end-1) - val(1:end-1);
    update = val(2:end) - val(1:end-1);
    
    valid = ~isnan(delta);
    delta = delta(valid);
    update = update(valid);
    T = numel(delta);

    % Design matrix: [delta, constant]
    X_glm = [delta, ones(T,1)];
    b = glmfit(X_glm, update, 'normal', 'constant', 'off');
    
    lr = b(1);
    block_effect = b(2);

    vars = struct( ...
    'vol', vol, ...
    'sto', sto, ...
    'val', val, ...
    'y_std', y_std, ...
    'y_mean', y_mean, ...
    'lr', lr, ...
    'block_effect', block_effect ...
    );

end

% LIKELIHOOD FUNCTION: Bernoulli predictive likelihood
function y = bernoulli_predictive(o, q, s)
    % Calculate the effective probability of observing 1.
    p = (1 - s) .* q + s .* (1 - q);
    
    % Compute the Bernoulli likelihood:
    %   If o == 1, likelihood is p; if o == 0, likelihood is (1-p).
    y = p .^ o .* (1 - p) .^ (1 - o);
end

% RESAMPLE FUNCTION: Particle resampling based on likelihood weights
function [idx, weights, resampled] = resample(likelihood, weights, config)
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

% HMM DIFFUSE FUNCTION: Diffusing the state probability
function q = hmm_diffuse(r, v)
    q = r .* (1 - v) + (1 - r) .* v;
end

% HMM UPDATE FUNCTION: Update particle filter based on observation likelihood
function r_new = hmm_update(o, s, q)
    % Compute the likelihood factor (ell) from the outcome and noise.
    ell = o .* (1 - s) + (1 - o) .* s;    
    
    % Update the state probability r_new using a Bayesian-like formula.
    % A small eps is added to avoid division by zero.
    r_new = ell .* q ./ (ell .* q + (1 - ell) .* (1 - q) + eps);
end

