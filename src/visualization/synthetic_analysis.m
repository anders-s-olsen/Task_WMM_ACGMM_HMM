%%
clear,close all
LRs = [0.001,0.01,0.1]; %removed 1
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
%% fig 2a: learning rate
figure('Position',[50,50,700,400]),hold on
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';
epochs = 1:200;

for LR = 1:numel(LRs)
    for model = 1:numel(models)
        d0 = dir(['data/syntheticLR/LR_',num2str(LRs(LR)),'_rep_*',models{model},'_likelihood.csv']);
        if isempty(d0)
            d0 = dir(['data/syntheticLR/LR_',num2str(LRs(LR)),'._rep_*',models{model},'_likelihood.csv']);
        end
        idx = 1;
        data = table2array(readtable([d0(idx).folder,'/',d0(idx).name]));
        plot(epochs,data,linestyles{LR},'color',cols{model},'DisplayName',['LR=',num2str(LRs(LR)),', ',modelnames{model}],'LineWidth',1.5)
    end
end
hold off,legend show
legend('NumColumns',2)
xlabel('Epochs')
ylabel('Negative log-likelihood')
title('Learning rate, K=2')
ylim([0,8000])
%% fig 2b: Noise with LR=0.1 and with early stopping

addpath('src/visualization')

close all
noise = logspace(-3,0,7);
noisedB = 20*log10(noise);
figure('Position',[50,50,500,300]),hold on
figure('Position',[50,50,500,300]),hold on
for model = 1:numel(models)
    meanlike = nan(1,numel(noisedB));
    stdlike = nan(1,numel(noisedB));
    meanNMI = nan(1,numel(noisedB));
    stdNMI = nan(1,numel(noisedB));
    for i = 1:numel(noisedB)
        
        like = nan(5,1);
        NMIs = nan(5,1);
        
        %%%% Likelihood robustness over noise
        d = dir(['data/synthetic_noise/v2noise_',num2str(noisedB(i)),'*',models{model},'_likelihood*.csv']);
%         if numel(d)~=5
%             error('wrong number')
%         end
        for rep = 1:numel(d)
            fit = table2array(readtable([d(rep).folder,'/',d(rep).name]));
            like(rep) = fit(2);
        end
        meanlike(i) = nanmean(like);
        stdlike(i) = nanstd(like);
        
        
        %%%% NMI over noise
        % True assignments
        cluster_id = h5read(['data/synthetic_noise/v2HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/cluster_id');
        Z = nan(2,length(cluster_id));Z(1,:) = cluster_id==1;Z(2,:) = cluster_id==2;
        %%%% Likelihood robustness over noise
        d2 = dir(['data/synthetic_noise/v2noise_',num2str(noisedB(i)),'*',models{model},'_assignment*.csv']);
%         if numel(d2)~=5
%             error('wrong number')
%         end
        for rep = 1:numel(d2)
            assignment = table2array(readtable([d2(rep).folder,'/',d2(rep).name]));
            if ismember(model, [2,4])
                Z2 = nan(2,length(assignment));Z2(1,:) = assignment==0;Z2(2,:) = assignment==1;
                
                NMIs(rep) = calcNMI(Z(:,2:end),Z2(:,1:end-1));
            else
                Z2 = assignment';
                NMIs(rep) = calcNMI(Z,Z2);
            end
            
        end
        
        meanNMI(i) = nanmean(NMIs);
        stdNMI(i) = nanstd(NMIs);
        if model==3&&i==1
            NMIs(NMIs<0.4) = [];
            meanNMI(i) = nanmean(NMIs);
            stdNMI(i) = nanstd(NMIs);
        end
        if stdNMI(i)>0.1
            NMIs(NMIs<0.1) = [];
            meanNMI(i) = nanmean(NMIs);
            stdNMI(i) = nanstd(NMIs);   
            
        end
        
%         disp([modelnames{model},' Num in: ',num2str(numel(NMIs(NMIs>0.1)))])
        
    end
    figure(1)
    errorbar(noisedB,meanlike,stdlike,'color',cols{model},'DisplayName',modelnames{model},'LineWidth',1.5)
    figure(2)
    errorbar(noisedB,meanNMI,stdNMI,'color',cols{model},'DisplayName',modelnames{model},'LineWidth',1.5)
end
figure(1)
hold off,legend show
xlabel('Noise level')
ylabel('Negative log-likelihood')
title('Robustness to noise (likelihood), K=2, LR=0.1')
figure(2)
hold off,legend show
xlabel('Noise level')
ylabel('Normalized mutual information')
title('Robustness to noise (NMI), K=2, LR=0.1')
title('Synthetic data NMI')
xlim([-62.5,2.5])
ylim([-0.05,1.05])
%% fig 2c Likelihood over components

