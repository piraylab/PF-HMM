function data = get_data_bin(experiment)

    currentFolder = fileparts(mfilename('fullpath'));
    cd(currentFolder);
    fdir = fullfile('..', 'mat_data', sprintf('experiment_%s', experiment));

    if strcmp('binA', experiment)
        fname = fullfile(fdir, 'data_binA.mat');
        S = load(fname);
        D = S.data;
        N = numel(D);
        data = cell(N,1);
    
        for i = 1:N
            subj = D{i};
    
            choice   = subj.choice; % 1/2
            outcome  = subj.outcome; % 1:2->0/1; 3:4->-1/0
        
            % Data preprocessing
            outcome(:,1:2) = 2*outcome(:,1:2) - 1; % 1:2->-1/1
            outcome(:,3:4) = 2*outcome(:,3:4) + 1; % 3:4->-1/1
            outcome(choice==2) = -outcome(choice==2);
            outcome(outcome==-1) = 0;
            choice(choice==2) = 0;
    
            out = struct();
            out.choice = choice;
            out.outcome = outcome;
            out.workerId = i;
    
            data{i} = out;
        end
    elseif strcmp('binB', experiment)
        fname = fullfile(fdir, 'dataset_matt2_all.mat');
        S = load(fname);
        data = S.data;
    else
        fprintf("Warning: experiment input either 'binA' or 'binB'. \n")
    end
end


