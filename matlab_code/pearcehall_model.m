function [val, tx, lr, signals] = pearcehall_model(parameters, outcomes, do_transform)
% PEARCEHALL_MODEL  Pearce-Hall model for binary outcomes and binary choices.
%
% outcomes: [T x B] matrix (0/1/NaN). Here this is data.outcome after masking by choice.
% val:      predicted P(choice==1) per trial (same size as outcomes)
%
% State:
%   V_t in [0,1]  : belief about P(outcome=1)
%   alpha_t in [0,1] : associability (learning rate), updated by unsigned PE
%
% Updates (when outcome observed):
%   delta_t = o_t - V_t
%   V_{t+1} = clip(V_t + alpha_t * delta_t, 0, 1)
%   alpha_{t+1} = clip((1-kappa)*alpha_t + kappa*abs(delta_t), 0, 1)
%
% Choice rule:
%   P(choice=1) = sigmoid( beta*(V_t - 0.5) + bias )
%
% parameters (unconstrained) -> transformed:
%   p1: V0_logit  -> V0 in (0,1)
%   p2: alpha0_logit -> alpha0 in (0,1)
%   p3: kappa_logit -> kappa in (0,1)
%   p4: beta_raw -> beta > 0 (softplus)
%   p5: bias (optional, fixed 0 if not provided)

    if nargin < 3, do_transform = 1; end
    if nargin < 2
        [data] = get_data('sealion', true, 1);
        outcomes = data{1}.outcome;
    end
    if nargin < 1, parameters = zeros(1,5); end

    if do_transform == 1
        eps_sigmoid = 1e-6;
        sigmoid = @(x) (1-eps_sigmoid)./(1+exp(-x)) + eps_sigmoid;
        softplus = @(x) log1p(exp(x)); % stable enough for typical ranges

        V0     = .5;
        alpha0 = 1;
        kappa  = sigmoid(parameters(1));

        tx = [V0, alpha0, kappa];
    else
        V0     = .5;
        alpha0 = 1;
        kappa  = parameters(1);
        tx = [V0, alpha0, kappa];
    end

    [T, num_blocks] = size(outcomes);

    val      = nan(T, num_blocks);
    Vmat     = nan(T, num_blocks);
    alphamat = nan(T, num_blocks);
    deltaMat = nan(T, num_blocks);
    updateMat= nan(T, num_blocks);

    for b = 1:num_blocks
        [val(:,b), Vmat(:,b), alphamat(:,b), deltaMat(:,b), updateMat(:,b)] = ...
            model_per_block_ph(V0, alpha0, kappa, outcomes(:,b));
    end

    signals = struct();
    signals.val   = val;
    signals.V     = Vmat;
    signals.alpha = alphamat;
    signals.delta = deltaMat;
    signals.update= updateMat;

    % 1) Pearce–Hall-native measure: mean associability per block
    alpha_blockmean = mean(alphamat, 1, 'omitnan'); % 1x4
    signals.alpha_blockmean = alpha_blockmean;

    % --- blockwise GLM (same lr analysis as your distr-HMM) ---
    % Relate update to delta with block-specific intercepts.
    % This produces a single slope per block (compact summary).
    delta  = outcomes - Vmat;  % (o - V) at each t (but Vmat stored at t)
    delta(end,:) = [];         % align with update (t -> t+1)
    update = Vmat(2:end,:) - Vmat(1:end-1,:);
    signals.delta_glm  = delta;
    signals.update_glm = update;

    y = [];
    x = [];
    s_const = [];

    for i = 1:size(delta,2)
        y = [y; update(:,i)];
        x = blkdiag(x, delta(:,i));
        s_const = blkdiag(s_const, ones(size(delta,1),1));
    end

    b_glm = glmfit([x s_const], y, 'normal', 'constant', 'off');

    lr = b_glm(1:num_blocks,:);           % slope per block (update ~ delta)
    signals.block_effect = b_glm(num_blocks+1:end,:); % intercept per block
    
end

function [predictions, Vhist, alphahist, deltahist, updatehist] = ...
    model_per_block_ph(V0, alpha0, kappa, o)

    T = length(o);

    p_choice1 = nan(T,1);
    Vhist     = nan(T,1);
    alphahist = nan(T,1);
    deltahist = nan(T,1);
    updatehist= nan(T,1);

    V = V0;
    a = alpha0;

    sigmoid = @(x) 1./(1+exp(-x));

    for t = 1:T
        % store current latent state
        Vhist(t)     = V;
        alphahist(t) = a;

        % choice probability is based on belief BEFORE seeing outcome
        predictions(t, 1) = V;

        if ~isnan(o(t))
            % prediction error w.r.t. outcome
            delta = o(t) - V;
            deltahist(t) = delta;

            V_new = V + a * delta;
            % V_new = min(max(V_new, 0), 1);

            updatehist(t) = V_new - V;

            % Pearce–Hall associability update (unsigned PE)
            a_new = (1 - kappa)*a + kappa*abs(delta);
            if a_new < 0 || a_new > 1
                fprintf("Pearce–Hall associability (alpha) update out-of-bounds");
            end

            V = V_new;
            a = a_new;
        end
    end
end
