function st = pfhmm_rho_bmc()
% PFHMM_RHO_BMC  Bayesian model comparison: PF-HMM vs PF-HMM+rho
    addpath('cbm');
    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    
    experiment = 'sealion';
    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));
    postfix = '';
    
    % Load LME for PF-HMM (no perseveration)
    fname_pf = fullfile(fdir, sprintf('pfhmm_fit_%s%s.mat', experiment, postfix));
    f = load(fname_pf);
    lme_pf = f.lme;

    % Load LME for PF-HMM + rho (perseveration)
    postfix = '';
    fname_pf_rho = fullfile(fdir, sprintf('pfhmm_rho_fit_%s%s.mat', experiment, postfix));
    f = load(fname_pf_rho);
    lme_pf_rho = f.lme;

    % Pairwise Bayesian Model Comparison
    % Columns: [PF-HMM, PF-HMM+rho]
    lme = [lme_pf, lme_pf_rho];

    [~, mf, ~, pxp, ~] = cbm_spm_BMS(lme);

    % Assemble output
    tbl = struct();
    tbl.rows = {'PF-HMM', 'PF-HMM+rho'};
    tbl.columns = {'ModelFrequency', 'ProtectedExceedanceProbability'};
    tbl.data = [mf(:), pxp(:)];

    st.table_struct = tbl;
    st.table = array2table(tbl.data, ...
        'VariableNames', tbl.columns, ...
        'RowNames', tbl.rows);

    fprintf('\n=== PF-HMM vs PF-HMM+rho Bayesian Model Comparison ===\n');
    disp(st.table);

end
