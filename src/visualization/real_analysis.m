%%
clear,close all

%% ACG K vs D + ACG scratch vs ACG full
close all
figure('Position',[100,100,800,300])
tiledlayout(1,2,'TileSpacing','compact','Padding','none')
nexttile
K = [1,4,7,10];

for k = 1:numel(K)
    d1 = dir(['data/real_ACG_initexperiment/K',num2str(K(k)),'ACG_MM_scratch*.csv']);
    tmp = nan(numel(d1),1);
    for r = 1:numel(d1)
        data = table2array(readtable([d1(r).folder,'/',d1(r).name]));
        tmp(r)=min(data);
    end
    likes(k,1) = mean(tmp);
    errors(k,1) = std(tmp);
    
    d2 = dir(['data/real_ACG_initexperiment/K',num2str(K(k)),'ACG_MM_Watsoninit*.csv']);
    tmp = nan(numel(d2),1);
    for r = 1:numel(d2)
        data = table2array(readtable([d2(r).folder,'/',d2(r).name]));
        tmp(r)=min(data);
    end
    likes(k,2) = mean(tmp);
    errors(k,2) = std(tmp);
end
b=bar(likes);hold on
b(1).FaceColor = [0.7,0.7,0.7];b(2).FaceColor = [0.3,0.3,0.3];
ngroups = size(likes, 1);
nbars = size(likes, 2);
% Calculating the width for each bar group
groupwidth = min(0.8, nbars/(nbars + 1.5));
for i = 1:nbars
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    er = errorbar(x, likes(:,i), errors(:,i));
    er.Color = [0 0 0];                            
    er.LineStyle = 'none';
end
xticks(1:4);
xticklabels({'1','4','7','10'});

xlabel('Number of components (K)')
ylabel('Negative log-likelihood')
% set(gca,'YDir','reverse')
legend('Random initialization','Watson initialization','Location','SouthWest')
ylim([-5.6*10^(5),-5*10^(5)])
yticks([-5.6:0.2:-5]*10^(5))

nexttile,hold on
K = [1,4,10];
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles = {'-','--',':'};
for m = 1:2%numel(models)
    for k = 1:numel(K)
        test_like = nan(50,5);
        d = dir(['data/real_K_D/K',num2str(K(k)),models{m},'_testlikelihood*.csv']);
        for i = 1:numel(d)
            test_like(:,i) = table2array(readtable([d(i).folder,'/',d(i).name]));
        end
        errorbar(1:2:50,nanmean(test_like(1:2:50,:),2),nanstd(test_like(1:2:50,:),[],2),'color',cols{m},'LineStyle',linestyles{k},'DisplayName',[modelnames{m},': K=',num2str(K(k))])
    end
    legend
end
xlabel('ACG rank (r)')
xticks(0:10:50)
% yticks([])
xlim([0,50])
% ylabel('Negative log-likelihood')

%% Likelihood plot over K
close all
figure('Position',[100,100,400,300]),
% subplot(2,1,1),
hold on
K = 1:10;
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'Rank-15 ACG-MM','Rank-15 ACG-HMM','Watson-MM','Watson-HMM'};
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
testlikes = nan(numel(K),5);
for m = 1:numel(models)
    testlikes = nan(10,5);
    for k = 1:numel(K)
        d = dir(['data/real_K/K',num2str(K(k)),models{m},'_testlikelihood*.csv']);
        for r = 1:numel(d)
            data = table2array(readtable([d(r).folder,'/',d(r).name]));
            testlikes(k,r) = data(1);
        end
    end
    
    errorbar(K,nanmean(testlikes,2),nanstd(testlikes,[],2),'color',cols{m},'DisplayName',modelnames{m},'LineWidth',1.5)
end
legend
xlim([0.7,10.3])
ylim([-5.6,-4.3]*10^5)
xlabel('Model order (K)')
ylabel('Negative test log-likelihood')
% title('Loss curves')

% %%%%%% Same for NMI
% subplot(2,1,2),hold on
% testNMIs = nan(numel(K),5);
% task_raw = table2array(readtable('data/raw/motor_ref.txt'));
% task = repmat(task_raw(1:120,:),29,1);
% task2 = repmat(task_raw(2:120,:),29,1);
% addpath('src/visualization')
% for m = 1:numel(models)
%     testNMIs = nan(10,5);
%     for k = 2:numel(K)
%         d = dir(['data/real_K/K',num2str(K(k)),models{m},'_assignment*.csv']);
%         for r = 1:numel(d)
%             data = table2array(readtable([d(r).folder,'/',d(r).name]));
%             
%             if m==1 || m==3
%                 testNMIs(k,r) = calcNMI(data',task');
%             else
%                 data2 = data(1:end-1,:);
%                 for kk = 1:k
%                     data3(kk,:) = data2(:)==kk-1;
%                 end
%                 testNMIs(k,r) = calcNMI(data3,task2');
%             end
%         end
%     end
%     
%     errorbar(K,nanmean(testNMIs,2),nanstd(testNMIs,[],2),'color',cols{m},'DisplayName',modelnames{m},'LineWidth',1.5)
% end
% legend
% xlim([1.5,10.5])
% xlabel('Model order (K)')
% ylabel('Normalized mutual information')
% return
% 
% K=10
% figure,subplot(2,1,1)
% plot(data2'+1.1*(1:K)),xlim([0,1000])
% subplot(2,1,2)
% plot(task+[0,1.1]),xlim([0,1000])

%% any of them?

task_raw = table2array(readtable('data/raw/motor_ref.txt'));
task = repmat(task_raw(1:120,:),29,1);
task2 = repmat(task_raw(2:120,:),29,1);
addpath('src/visualization')
for m = 1:numel(models)
    
    for k = 2:numel(K)
        testNMIs = nan(10,5,k);
        d = dir(['data/real_K/K',num2str(K(k)),models{m},'_assignment*.csv']);
        for r = 1:numel(d)
            data = table2array(readtable([d(r).folder,'/',d(r).name]));
            
            if m==1 || m==3
                for kk = 1:k
                testNMIs(k,r,kk) = calcNMI(data(:,kk)',task');
                end
            else
                data2 = data(1:end-1,:);
                for kk = 1:k
                    data3 = data2(:)==kk-1;
                    testNMIs(k,r,kk) = calcNMI(data3,task2');
                end
                
            end
        end
    end
    
end


%% regu, K, LR experiment, test likelihood

K = [2,4,8];
regus = [1e-06,1e-05,0.0001,0.001,0.01,0.1];
regus_str = {'1e-06','1e-05','00001','0001','001','01'};
LRs = [0.01,0.1,1,10];
LRs_str = {'001','01','10','100'};

cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';linestyles{5} = '--o';

figure('Position',[100,100,1500,800])
for k = 1:numel(K)
    for LR = 1:numel(LRs)
        alllikes = nan(numel(regus),5);
        for regu = 1:numel(regus)
            d = dir(['data/real_K_regu_LR/K',num2str(K(k)),'regu',regus_str{regu},'LR',LRs_str{LR},'ACG_MM_testlikelihood_rep*.csv']);
            for r = 1:numel(d)
                data = table2array(readtable([d(r).folder,'/',d(r).name]));
                alllikes(regu,r) = data(1);
            end
        end
        likemean = nanmean(alllikes,2);
        likestd = nanstd(alllikes,[],2);
        errorbar(log10(regus),likemean,likestd,'color',cols{LR},'LineStyle',linestyles{k},'LineWidth',1.5,'DisplayName',['LR= ',num2str(LRs(LR)),', K=',num2str(K(k))])
        hold on
    end
end
legend

%% regu, K, LR experiment, full likelihood

K = [2,4,8];
regus = [1e-06,1e-05,0.0001,0.001,0.01,0.1];
regus_str = {'1e-06','1e-05','00001','0001','001','01'};
LRs = [0.01,0.1,1,10];
LRs_str = {'001','01','10','100'};

cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';linestyles{5} = '--o';


for k = 1:numel(K)
    figure('Position',[100,100,1500,800])
    c=1;
    for LR = 1:numel(LRs)
        for regu = 1:numel(regus)
            alllikes = nan(5,10000);
            d = dir(['data/real_K_regu_LR/K',num2str(K(k)),'regu',regus_str{regu},'LR',LRs_str{LR},'ACG_MM_likelihood_rep*.csv'])
            for r = 1:numel(d)
                data = table2array(readtable([d(r).folder,'/',d(r).name]));
                alllikes(r,:) = data;
            end
            subplot(numel(LRs),numel(regus),c)
            errorbar(1:10000,nanmean(alllikes),nanstd(alllikes),'LineWidth',1.5)
            c = c + 1;
        end
    end
end



%% check fit, ACG
close all
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
m=1;K=8;
likes_onemodel = nan(1,5);
d = dir(['data/real_K/K',num2str(K),models{m},'_testlikelihood*.csv']);
for i = 1:numel(d)
    data = table2array(readtable([d(i).folder,'/',d(i).name]));
    likes_onemodel(i) = data(1);
end
[~,idx] = nanmin(likes_onemodel);

figure('Position',[100,100,1700,900])
tiledlayout(3,K)
nexttile(1,[1,K])
assignment = table2array(readtable(['data/real_K/K',num2str(K),models{m},'_assignment',num2str(idx-1),'.csv']));
plot(assignment+[0:K-1]*1.1)
ylim([-0.1,K*1.1+0.1])
xticks([])

task_raw = table2array(readtable('data/raw/motor_ref.txt'));
task = repmat(task_raw(1:120,:),29,1);
nexttile(K+1,[1,K])
plot(task(:,1),'LineWidth',1.5,'color',[0,0.5,0]),hold on
plot(task(:,2)+0.2+max(task(:)),'LineWidth',1.5,'color',[0.5,0,0])
title('Right/left hand motor task')
% xlabel('Time [min]'),
ylim([-.3,2.7])
% xlim([-.1 4.1])
% xticks(0:4)
yticks([mean(task(:,1)),mean(task(:))+max(task(:))+0.2])
yticklabels({'RH','LH'})

for kk = 1:K
    b = table2array(readtable(['data/real_K/K',num2str(K),models{m},'_comp',num2str(kk-1),'_',num2str(idx-1),'.csv']));
    nexttile(K*2+kk)
    imagesc(b),colorbar,axis square
    load('data/external/labels_cell_100.mat')
    yticks(1:100);yticklabels((labels));
%     xticks(1:100);xticklabels((labels));
    set(gca,'FontSize',7)
    title(['Component ',num2str(kk)])
end
%% check NMI
addpath('/dtu-compute/macaroni/Task_WMM_ACGMM_HMM/src/visualization')
PI = (task-min(task))./sum(task-min(task),2);
calcNMI(PI',assignment');
calcNMI(PI(2:end,:)',assignment(1:end-1,:)');

%% Check components, ACG
close all
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
m=1;K=6;
likes_onemodel = nan(1,5);
d = dir(['data/real_K/K',num2str(K),models{m},'_testlikelihood*.csv']);
for i = 1:numel(d)
    data = table2array(readtable([d(i).folder,'/',d(i).name]));
    likes_onemodel(i) = data(1);
end

[~,idx] = nanmin(likes_onemodel);
for kk = 1:K
    b = table2array(readtable(['data/real_K/K',num2str(K),models{m},'_comp',num2str(kk-1),'_',num2str(idx-1),'.csv']));
    figure('Position',[100,100,1000,1000]),
    imagesc(b),colorbar,axis square
    load('data/external/labels_cell_100.mat')
    yticks(1:100);yticklabels((labels));
    xticks(1:100);xticklabels((labels));
    set(gca,'FontSize',7)
end

%% Check components, watson
close all
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
m=4;K=5;
likes_onemodel = nan(1,5);
d = dir(['data/real_K/K',num2str(K),models{m},'_likelihood*.csv']);
for i = 1:numel(d)
    data = table2array(readtable([d(i).folder,'/',d(i).name]));
    likes_onemodel(i) = data(1);
end

[~,idx] = nanmin(likes_onemodel);
figure
for kk = 1:K
    b = table2array(readtable(['data/real_K/K',num2str(K),models{m},'_comp',num2str(kk-1),'_mu',num2str(idx-1),'.csv']));
    subplot(1,K,kk),barh(1:100,b)
    load('data/external/labels_cell_100.mat')
    yticks(1:100);yticklabels((labels));
    set(gca,'FontSize',7)
end


%% Step 1: Ensure good regularization factor for ACG.
% Likelihood vs regularization and K, ACGMM
% Note that regularization factor appears to be independent of K


regus = fliplr([1e-08,1e-07,1e-06,1e-05,1e-04,1e-03,0.00215,0.00464,1e-02,0.0215,0.0464,1e-01,1e-00]);
regus_str = fliplr({'1e-08','1e-07','1e-06','1e-05','00001','0001','000215','000464','001','00215','00464','01','10'})
K = 1:10;

alllikes = nan(numel(K),numel(regus));
for regu = 1:numel(regus)
    for k = K
        d = dir(['data/real_K_LR/K',num2str(k),'regu',regus_str{regu},'.csv']);
        try
        data = table2array(readtable([d.folder,'/',d.name]));
        alllikes(k,regu) = data(1);
        catch,end
    end
end

close all
figure('Position',[100,100,1700,600]),
idx = [1,2,5,8,9,10,11,12,13];
subplot(2,2,1),semilogx(regus(idx),alllikes(:,idx),'k-o','LineWidth',1.5)
grid on
subplot(2,2,2),imagesc(fliplr(alllikes)),colorbar
xlabel('regularization')
ylabel('num components')
xticks(1:numel(regus)),
colormap(flipud(jet))
xticklabels({'1e-08','1e-07','1e-06','1e-05','1e-04','1e-03','1e-02','1e-01','1'})
xticklabels({'10^{-8}','10^{-7}','10^{-6}','10^{-5}','10^{-4}','10^{-3}','0.00215','0.00464',...
    '10^{-2}','0.0215','0.0464','10^{-1}','1'})
subplot(2,2,3)
plot(K,alllikes(:,regus==0.001));
xlabel('Number of components K')
ylabel('Negative log likelihood')


%% Step 2: Ensure reasonable learning rate. fig 3a: Likelihood over LR
figure('Position',[50,50,700,400]),hold on
LRs = [0.001,0.01,0.1,1,10]; %removed 1
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
linestyles{1} = '-';linestyles{2} = '--';linestyles{3} = ':';linestyles{4} = '-.';linestyles{5} = '--o';
epochs = 1:200;

maxK = 6;

for model = 1:numel(models)
    meanlike = nan(1,maxK);
    for LR = 1:numel(LRs)
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

%0.1 for watson and ACG HMM, 0.01 for ACG MM

%% Likelihood over K
figure('Position',[50,50,700,400]),hold on
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
likes = nan(10,4,5);
for m = 1:numel(models)
    for K = 1:10
        d = dir(['data/real_K/K',num2str(K),models{m},'_likelihood*.csv']);
        for i = 1:numel(d)
            data = table2array(readtable([d(i).folder,'/',d(i).name]));
            likes(K,m,i) = data(1);
        end
        likes_mean = nanmean(likes,3);
        likes_std = nanstd(likes,[],3);
    end
    errorbar(1:10,likes_mean(:,m),likes_std(:,m),'color',cols{m},'DisplayName',modelnames{m},'LineWidth',1.5),hold on
end
legend


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













