%%
clear,close all
LRs = [0.001,0.01,0.1]; %removed 0.001
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};

%% fig 2a: learning rate
figure('Position',[50,50,700,400]),hold on
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';
epochs = 1:200;

for LR = 1:numel(LRs)
    for model = 1:numel(models)
        d = dir(['data/syntheticLR/LR_',num2str(LRs(LR)),'_rep_*',models{model},'_likelihood.csv']);
        if isempty(d)
            d = dir(['data/syntheticLR/LR_',num2str(LRs(LR)),'._rep_*',models{model},'_likelihood.csv']);
        end
        idx = 1;
        data = table2array(readtable([d(idx).folder,'/',d(idx).name]));
        plot(epochs,data,linestyles{LR},'color',cols{model},'DisplayName',['LR=',num2str(LRs(LR)),', ',modelnames{model}],'LineWidth',1.5)
    end
end
hold off,legend show
legend('NumColumns',2)
xlabel('Epochs')
ylabel('Negative log-likelihood')
title('Learning rate, K=2')
%% fig 2b: Noise with LR=0.1 and with early stopping

noise = logspace(-4,-1,9);
noisedB = 20*log10(noise);
figure('Position',[50,50,700,400]),hold on

for model = 1:numel(models)
    meanlike = nan(1,5);
    stdlike = nan(1,5);
    for i = 1:numel(noisedB)
        d = dir(['data/synthetic_noise/noise_',num2str(noisedB),'*',models{model},'_likelihood*.csv']);
        if numel(d)~=5
            error('wrong number')
        end
        for rep = 1:5
            fit = table2array(readtable([d(rep).folder,'/',d(rep).name]));
            like(rep) = fit(2);
        end
        
        meanlike(i) = mean(like);
        stdlike(i) = std(like);
    end
    errorbar(noisedB,meanlike,stdlike,'DisplayName',modelnames{model})
end
hold off,legend show
xlabel('Noise level')
ylabel('Negative log-likelihood')
title('Robustness to noise, K=2, LR=0.1')

%% fig 2c Likelihood over components

