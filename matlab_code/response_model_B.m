function out = response_model_B(p1_pred, trialvalue, choice01, params)
% Value-based asymmetric response model (Dataset B)

    beta = exp(params.temp);   % inverse temperature

    % Expected utility
    p = p1_pred;
    U = p .* trialvalue + (1-p) .* (-10);

    z = beta .* U;
    % f = 1 ./ (1 + exp(-z));
    % standard stable logistic
    f = zeros(size(z));
    idx = z >= 0;
    f(idx)  = 1 ./ (1 + exp(-z(idx)));
    f(~idx) = exp(z(~idx)) ./ (1 + exp(z(~idx)));


    y = choice01;

    % mask invalid trials
    valid = ~isnan(y) & ~isnan(f);
    
    if ~any(valid(:))
        loglik = -1e16;
    else
        eps0 = 1e-12;
        loglik = sum( ...
            y(valid).*log(f(valid) + eps0) + ...
            (1-y(valid)).*log(1 - f(valid) + eps0) ...
        );
    end

    out = struct();
    out.loglik = loglik;
    out.p_choice = f;
    out.beta = beta;
end