%%
clear,close all
LRs = [0.01,0.1,1,10,100]; %removed 1
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};

%% fig 3a: Likelihood over LR
figure('Position',[50,50,700,400]),hold on
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';linestyles{5} = '--o';
epochs = 1:200;

maxK = 6;

for model = 1:numel(models)
    meanlike = nan(1,maxK);
    for LR = 1:numel(LRs)-1
        d0 = dir(['data/realLR/LR_',num2str(LRs(LR)),'_',models{model},'_likelihood.csv']);
        if isempty(d0)
            d0 = dir(['data/realLR/LR_',num2str(LRs(LR)),'._',models{model},'_likelihood.csv']);
        end
        data = table2array(readtable([d0.folder,'/',d0.name]));
        plot(data,linestyles{LR},'color',cols{model},'DisplayName',['LR=',num2str(LRs(LR)),', ',modelnames{model}],'LineWidth',1.5)
        
    end
end
hold off,legend show
legend('NumColumns',2)
xlabel('Epochs')
ylabel('Negative log-likelihood')
title('Learning rates')
% ylim([0,8000])

% ACG MM: 0.01, same for hmm
% Watson: 1

%% Check components, watson

a = table2array(readtable('data/real_K/K5Watson_MM_assignment0.csv'));
figure,plot(a+[1:5])

for kk = 1:5
b = table2array(readtable(['data/real_K/K5Watson_HMM_comp',num2str(kk-1),'_mu0.csv']));
figure,barh(1:200,b)
load('data/external/labels_cell.mat')
yticks(1:200);yticklabels(labels);
set(gca,'FontSize',7)
end
%%
close all

for K = 1:10
    like = nan(5,1);
    for r = 1:5
        data = table2array(readtable(['data/real_K/K',num2str(K),'ACG_MM_likelihood',num2str(r-1),'.csv']));
        like(r) = data(2);
    end
    [~,best_like] = min(like);
    comp = table2array(readtable(['data/real_K/K',num2str(K),'ACG_MM_comp0',num2str(best_like-1),'.csv']));
    figure,imagesc(inv(comp*comp'))
end

%% check components
K=3;
close all

P = 200;
Cp = gammaln(P/2)-log(2)-P/2*log(pi);
figure('Position',[50,50,1500,500]),
for r = 1:5
    comp = table2array(readtable(['data/real_K/K',num2str(K),'ACG_MM_comp0',num2str(r-1),'.csv']));
    comp=comp;
    disp(det(comp))
    figure('Position',[50,50,1500,500]),
    subplot(1,3,1),
    imagesc(comp),colorbar,axis square
    subplot(1,3,2),
    imagesc(comp*comp'),colorbar,axis square
    subplot(1,3,3)
    imagesc(inv(comp*comp')),colorbar,axis square
%     logdeta = -2*sum(log(abs(diag(comp))));
%     for i = 1:P
%         tmp = zeros(1,P);
%         tmp(i) = 1;
%     
%         B1 = tmp*comp;B1 = sum(B1.*B1,2);
%         val(i) = Cp-0.5*logdeta+(-P/2)*log(B1);
%     end
%     subplot(1,5,r)
%     barh(val)
    
    
end



%% Fig 3a: NMI best match with motor task

figure('Position',[50,50,700,400]),hold on
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';
epochs = 1:200;

motor = table2array(readtable('data/raw/motor_ref.txt'));
motor = motor-min(motor);motor = motor./max(motor);

for model = 1:numel(models)
    meanlike = nan(1,maxK);
    for K = 2:maxK
        d0 = dir(['data/synthetic_K/K',num2str(K),models{model},'_likelihood*.csv']);
        
        like = nan(1,3);
        for i = 1:3
            data = table2array(readtable([d0(i).folder,'/',d0(i).name]));
            like(i) = data(2);
        end
        [~,best] = min(like);
        
        assignment = table2array(readtable([d0(best).folder,'/K',num2str(K),models{model},'_assignment',num2str(best),'.csv']));
        Z = nan(size(motor,1),K);
        for sub = 1:29
            if ismember(model, [2,4])
                for k = 1:K
                    Z = assignment(:,sub)==k;
                    rightNMI(k) = calcNMI(motor(2:end,1)',Z(1:end-1)');
                    leftNMI(k) = calcNMI(motor(2:end,2)',Z(1:end-1)');
                end
                maxrightNMI(K,sub) = max(rightNMI);
                maxleftNMI(K,sub) = max(leftNMI);
                
            else
                for k = 1:K
                    assignment = reshape(assignment,240,29,2);
                    Z = squeeze(assignment(:,sub,:));
                    rightNMI(k) = calcNMI(motor(:,1)',Z');
                    leftNMI(k) = calcNMI(motor(:,2)',Z');
                end
                maxrightNMI(K,sub) = max(rightNMI);
                maxleftNMI(K,sub) = max(leftNMI);
            end
        end
        
        meanmaxright = mean(maxrightNMI,2);
        meanmaxleft = mean(maxleftNMI,2);
        stdmaxright = std(maxrightNMI,2);
        stdmaxleft = std(maxleftNMI,2);
        
    end
    plot(1:maxK,meanmaxright,stdmaxright,'color',cols{model},'DisplayName',['right:',modelnames{model}],'LineWidth',1.5)
    plot(1:maxK,meanmaxleft,stdmaxleft,'color',cols{model},'DisplayName',['left:',modelnames{model}],'LineWidth',1.5)
end
hold off,legend show
xlabel('K')
ylabel('Negative log-likelihood')
title('Loss curves')
% ylim([0,8000])













