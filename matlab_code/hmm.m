function [predictions, b] = hmm(outcome, v, s)
% HMM Implements a basic Hidden Markov Model update procedure.
%
%   [predictions, b] = hmm(outcome, v, s)
%
%   Inputs:
%       outcome - a T x num_dim matrix of observed outcomes, where each row
%                 represents a trial and each column represents a different
%                 condition. The outcomes are assumed to follow a noisy 
%                 process relative to the hidden state.
%       v       - probability of switching the hidden state. This parameter
%                 influences the prior update.
%       s       - noise parameter indicating how closely outcomes follow the
%                 hidden state. For s = 0, outcomes perfectly reflect the state.
%
%   Outputs:
%       predictions - a T x num_dim matrix containing the prior predictions of
%                     the hidden state for each trial.
%       b           - coefficients from a generalized linear model (GLM)
%                     that relates the prediction error (delta) to the update in
%                     belief, computed over each block (dimension).
%
%   The function iteratively updates the belief about the hidden state based on
%   the outcome at each time point using a Bayesian-like rule. It optionally fits
%   a GLM to the update data to estimate a learning rate per block.

    % Determine the number of trials (T) and the number of dimensions (num_dim)
    [T, num_dim] = size(outcome);
    
    % Initialize the belief state 'r' as 0.5 (neutral probability) for all dimensions.
    r = 0.5 * ones(1, num_dim);
    
    % Preallocate the matrices to store predictions, prediction errors (delta),
    % and updates in the belief for each trial.
    predictions = nan(T, num_dim);
    delta = nan(T, num_dim);
    update = nan(T, num_dim);
    
    % Loop over each trial to update predictions based on observed outcomes.
    for t = 1:T
        % Store the current belief 'r' as the prediction for the current trial.
        predictions(t, :) = r;
        
        % Compute the prior probability 'q' that incorporates the chance of switching.
        % This is a weighted sum of the current belief and its complement.
        q = r .* (1 - v) + (1 - r) .* v;
        
        % Extract the outcome for the current trial.
        o = outcome(t, :);
        
        % Assign the updated belief to the prior q.
        r_new = q;
        
        % If the outcome is valid (i.e., not NaN), update the belief.
        if ~isnan(o)
            % Calculate the prediction error: difference between the observed outcome and
            % the current prediction (belief).
            delta(t, :) = o - r;
            
            % Compute the likelihood ('ell') of observing the outcome given the noise.
            % When o equals 1, ell is (1-s); when o equals 0, ell is s.
            ell = o .* (1 - s) + (1 - o) .* s;
            
            % Apply a Bayesian update rule to compute the new belief.
            % The numerator weights the prior q by the likelihood of the outcome,
            % while the denominator normalizes the value.
            r_new = ell .* q ./ (ell .* q + (1 - ell) .* (1 - q));
            
            % Record the magnitude of the update (i.e., the change in belief).
            update(t, :) = r_new - r;
        end
        
        % Set the updated belief for use in the next iteration.
        r = r_new;
    end

    % If a second output argument is requested, perform further analysis by fitting a GLM.
    if nargout > 1
        % Initialize empty matrices for accumulating GLM data.
        y = [];        % Dependent variable: concatenated belief updates (learning signal).
        x = [];        % Independent variable: block-diagonal matrix of prediction errors.
        s_const = [];  % Constant term (ones) for each trial (used as an intercept in GLM).
        
        % Loop through each dimension to compile the data for GLM.
        for i = 1:num_dim
            % Concatenate the update values for dimension i into the response vector.
            y = [y; update(:, i)];
            
            % Build a block-diagonal matrix for delta (prediction error) for each dimension.
            x = blkdiag(x, delta(:, i));
            
            % Build a block-diagonal matrix of ones (acting as the constant term)
            % for each trial within the current dimension.
            s_const = blkdiag(s_const, ones(T, 1));
        end
        
        % Fit a Generalized Linear Model (GLM) to relate the prediction error (delta) to
        % the update in belief. Here, the combined matrix [x s_const] is used as the design
        % matrix, 'y' as the response variable, and a normal distribution is assumed.
        % The 'constant','off' option is set because the constant is already included in s_const.
        b = glmfit([x s_const], y, 'normal', 'constant', 'off');
    end

end
