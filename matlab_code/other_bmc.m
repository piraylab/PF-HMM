function st = other_bmc(experiment, postfix) 
% OTHER_BMC  Bayesian Model Comparison (BMS)

    if nargin<2
        experiment = "sealion"; % 'sealion', 'turtle', 'binA', 'binB'
        postfix = ''; % ''
    end

    addpath('cbm');
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    
    % Alternate models
    models = {'pearcehall_fmincon', 'binary_hgf_fmincon'}; % , 'dbd1_fmincon'
    
    % Load PFHMM LME
    fname_pf = fullfile(fdir, sprintf('pfhmm_rho_fit_%s%s.mat', experiment, postfix));
    fprintf('Loading file: %s \n', fname_pf);
    f = load(fname_pf);
    lme_pf = f.lme;
    
    % Preallocate 
    nAlt = numel(models);
    pair_data = nan(nAlt + 1, 3);    % [MF_pair_alt, MF_pair_pf, PXP_pair_pf]
    lme_alt_all = [];  % will become [Nsub x nAlt]

    % Pairwise comparisons
    for i=1:nAlt
        fname = fullfile(fdir, sprintf('%s_%s.mat', models{i}, experiment));
        fprintf('Loading file: %s \n', fname);
        f = load(fname);

        % Store alternative LMEs for later global BMS
        if isempty(lme_alt_all)
            lme_alt_all = nan(numel(f.lme), nAlt);
        end
        lme_alt_all(:, i) = f.lme;

        % Pairwise BMS between ALT and PFHMM (2 columns)
        tbl_pair_i = bmc_run(lme_pf, f.lme);

        pair_data(i, :) = tbl_pair_i.data;   % [MF_pair_alt, MF_pair_pf, PXP_pair_pf]
    end
    
    % Global BMS: compare all models together
    lme_global = [lme_alt_all, lme_pf];  % [ALT1 ALT2 ... PF]
    [~, mf_global, ~, pxp_global, ~] = cbm_spm_BMS(lme_global);

    % mf_global and pxp_global are length (nAlt+1). We stack them into 2 columns.
    global_data = [mf_global(:), pxp_global(:)]; % [MF_global, PXP_global]
    
    % Assemble final table struct
    tbl = struct();
    tbl.rows = [models(:); {'PF'}];
    tbl.columns = { ...
        'MF_pair_alt', ...   % pairwise model frequency of alternative (ALT vs PF)
        'MF_pair_pf',  ...   % pairwise model frequency of PFHMM (ALT vs PF)
        'PXP_pair_pf', ...   % pairwise PXP that PFHMM is most frequent (ALT vs PF)
        'MF',   ...   % global model frequency across all models
        'PXP'   ...   % global protected exceedance probability across all models
    };
    tbl.data = [pair_data, global_data];

    % Output + display 
    st.table_struct = tbl;
    T = array2table(tbl.data, 'VariableNames', tbl.columns, 'RowNames', tbl.rows);
    st.table = T;
    fprintf('\n=== Bayesian Model Comparison Results ===\n');
    fprintf('Pairwise columns (MF_pair_*, PXP_pair_pf): each ALT compared against PFHMM only.\n');
    fprintf('Global columns (MF_global, PXP_global): all models compete simultaneously.\n\n');
    disp(T);

end

function tbl = bmc_run(lme_pf, other_lme)
    % BMC_PAIRWISE_PF  Pairwise BMS between [ALT vs PFHMM].
    
    lme = [other_lme lme_pf];
    
     % cbm_spm_BMS returns MF and PXP for each column/model
    [~, mf, ~, pxp, ~] = cbm_spm_BMS(lme);
    
    % mf(1)=ALT, mf(2)=PF; pxp(2)=prob PF is most frequent (protected)
    tbl.data = [mf(1), mf(2), pxp(2)];
    tbl.columns = {'MF_pair_alt', 'MF_pair_pf', 'PXP_pair_pf'};
end