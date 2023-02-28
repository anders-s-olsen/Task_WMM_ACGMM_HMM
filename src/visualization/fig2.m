%%
clear,close all
LRs = [0.001,0.01,0.1]; %removed 1
models = {'ACG_MM','ACG_HMM','Watson_MM','Watson_HMM'};
modelnames = {'ACG-MM','ACG-HMM','Watson-MM','Watson-HMM'};
cols{1} = [0,0.5,0];cols{2} = [0.5,0,0];cols{3} = [0,0,0.5];cols{4} = [0.5,0,0.5];
cols{1} = [0,0.4,0];cols{2} = [0,0.7,0];cols{3} = [0.4,0,0];cols{4} = [0.8,0,0];
cols{1} = [0,0,0.5];cols{2} = [0.5,0,0.5];cols{3} = [0,0.3,0];cols{4} = [0,0.6,0];
addpath('src/visualization')

% figure('Position',[100,100,720,500])
% tiledlayout(2,2,'TileSpacing','compact','Padding','none')


%% fig A: Synth.

noise = logspace(-3,0,7);%noise(end)=[];
noisedB = 20*log10(noise);
% nexttile,hold on
figure('Position',[100,100,500,250]),
tiledlayout(2,1,'TileSpacing','compact','Padding','none')
hold on
for model = 1:numel(models)
    meanlike = nan(1,numel(noisedB));
    stdlike = nan(1,numel(noisedB));
    meanNMI = nan(1,numel(noisedB));
    stdNMI = nan(1,numel(noisedB));
    for i = 1:numel(noisedB)
        
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
    errorbar(noisedB,meanNMI,stdNMI,'color',cols{model},'DisplayName',modelnames{model},'LineWidth',1.5)
end

hold off,legend show
% legend('Location','SouthWest')
xlabel('Noise level')
ylabel('Normalized mutual information')
title('Robustness to noise (NMI), K=2, LR=0.1')
title('Synthetic data robustness to noise')
xlim([-62.5,2.5])
ylim([-0.05,1.05])
exportgraphics(gcf,'reports/figures/fig2_1.png','Resolution','300')

%% ACG K vs D + ACG scratch vs ACG full
% nexttile
figure('Position',[100,100,500,500])
tiledlayout(2,1,'TileSpacing','compact','Padding','none')
K = [1,4,7,10];
nexttile
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
ylabel('Test nll')
% set(gca,'YDir','reverse')
legend('Random initialization','Watson initialization','Location','SouthWest')
ylim([-5.6*10^(5),-5*10^(5)])
yticks([-5.6:0.2:-5]*10^(5))
title('Rank-1 ACG initialization')

nexttile,hold on
K = [1,4,10];
linestyles = {'-','--',':'};
for m = 1:2%numel(models)
    for k = 1:numel(K)
        test_like = nan(50,5);
        d = dir(['data/real_K_D/K',num2str(K(k)),models{m},'_testlikelihood*.csv']);
        for i = 1:numel(d)
            test_like(:,i) = table2array(readtable([d(i).folder,'/',d(i).name]));
        end
        %         errorbar(1:2:50,nanmean(test_like(1:2:50,:),2),nanstd(test_like(1:2:50,:),[],2),'color',cols{m},'LineStyle',linestyles{k},'DisplayName',[modelnames{m},': K=',num2str(K(k))])
        plot(1:50,nanmean(test_like,2),'color',cols{m},'LineStyle',linestyles{k},'DisplayName',[modelnames{m},': K=',num2str(K(k))],'LineWidth',1.5)
    end
    legend
end
xlabel('ACG rank (r)')
xticks(0:10:50)
% yticks([])
xlim([0,50])
title('ACG rank')
ylabel('Test nll')
exportgraphics(gcf,'reports/figures/fig2_2.png','Resolution','300')

%% Likelihood plot over K
% nexttile
figure('Position',[100,100,500,250])
% subplot(2,1,1),
hold on
K = 1:10;
testlikes = nan(numel(K),5);
modelnames = {'Rank-15 ACG-MM','Rank-15 ACG-HMM','Watson-MM','Watson-HMM'};
for m = 1:numel(models)
    testlikes = nan(10,5);
    for k = 1:numel(K)
        d = dir(['data/real_K/K',num2str(K(k)),models{m},'_testlikelihood*.csv']);
        for r = 1:numel(d)
            data = table2array(readtable([d(r).folder,'/',d(r).name]));
            if m==1||m==2
                testlikes(k,r) = data(end);
            else
                testlikes(k,r) = data(1);
            end
        end
    end
    if m==1 || m==2
        yyaxis left
    else
        yyaxis right
    end
    errorbar(K,nanmean(testlikes,2),nanstd(testlikes,[],2),'color',cols{m},'DisplayName',modelnames{m},'LineWidth',1.5,'LineStyle','-')
end
legend
xlim([0.7,10.3])
% ylim([-5.6,-4.3]*10^5)
xlabel('Model order (K)')
title('Model order')

yyaxis left
ylim([-5.9,-5.7]*10^5)
ylabel('ACG test nll')
yticks([-5.9:0.1:-5.7]*10^5)
yyaxis right
ylim([-5.75,-4.2]*10^5)
ylabel('Watson test nll')
yticks([-5.5:0.5:-4.5]*10^5)

ax = gca;
ax.YAxis(1).Color = [0.25,0,0.5];
ax.YAxis(2).Color = [0,0.45,0];
exportgraphics(gcf,'reports/figures/fig2_3.png','Resolution','300')
%%
% exportgraphics(gcf,'reports/figures/fig2.png','Resolution','300')

%%
addpath(genpath('BrainNetViewer'))
close all

% states = dir('data/real_K/K5ACG_MM_comp*0.csv')
% states = dir('data/nodes_edges/K3ACG_MM_comp*0.csv');

for m = [1]
    d = dir(['data/real_fit/K4',models{m},'_likelihood*.csv'])
    likes = nan(1,5);
    for r = 1:numel(d)
        data = table2array(readtable([d(r).folder,'/',d(r).name]));
        likes(r) = min(data);
    end
    [~,best] = nanmin(likes);
    if m==1
        states = dir(['data/real_fit/K4',models{m},'_comp*_',num2str(best-1),'.csv'])
    elseif m==3
        states = dir(['data/real_fit/K4',models{m},'_comp*_mu',num2str(best-1),'.csv'])
    end
    for i = 1:numel(states)
        
        
        
        if m==3
            mu = table2array(readtable([states(i).folder,'/',states(i).name]));
            if sum(mu>0)>sum(mu<0)
                mu = -mu;
            end
            kap = table2array(readtable([states(i).folder,'/',states(i).name(1:end-7),'kappa',num2str(best-1),'.csv']));
            A = sqrt(kap).*mu*mu';
        elseif m==1
            A = table2array(readtable([states(i).folder,'/',states(i).name]));
            A = A*A'+eye(100);
        end
        
        A_diag = diag(A);
        
        if m==1
            
            nodes = readtable('Node_Schaefer100.node','FileType','text');
            nodes.Var5 = A_diag./max(A_diag);
            %     nodes.Var4(Mpos) = 2;
            %     nodes.Var4(Mneg) = 1;
            nodes.Properties.VariableNames = {'# Schaefer_100','2','3','4','5','6'};
            writetable(nodes,['data/nodes_edges/nodes_state',num2str(i),'.node'],'FileType','text','Delimiter','\t')
            
            A2 = A;
            A2(A2==diag(diag(A2))) = nan; % set diag to zero
            A2 = A2-min(A2(:));
            A2 = A2/max(A2(:));
            A2 = A2./max(abs(A2(:))); %max is one
            
            q1 = quantile(A2(A2~=0),0.025);
            q2 = quantile(A2(A2~=0),0.975);
            A2(A2>q1&A2<q2) = 0;
            A2(A2<0|A2>0) = A2(A2<0|A2>0)-0.5;
            A2(isnan(A2)) = 0;
            
            %         A2(abs(A2(:))<0.9) = 0; %all less than 0.9 out
            %         A2(abs(A2(:))<quantile(abs(A2(:)),0.90)) = 0;
            edges = array2table(A2);
            writetable(edges,['data/nodes_edges/edges_state',num2str(i),'.edge'],'FileType','text','Delimiter','\t','WriteVariableNames',false)
            
            % make edge plot
            %         BrainNet_MapCfg('BrainMesh_ICBM152.nv',...
            %             ['data/nodes_edges/nodes_state',num2str(i),'.node'],...
            %             ['data/nodes_edges/edges_state',num2str(i),'.edge'],...
            %             'data/nodes_edges/options.mat',...
            %             ['data/nodes_edges/',models{m},'_state_edge',num2str(i),'.jpg'])
        end
        if m==3
            V = niftiread('data/external/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii');
            info = niftiinfo('data/external/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii');
            info.Datatype = 'double';
            V2 = zeros(size(V));
            A_diag = A_diag./norm(A_diag);
            for r = 1:numel(unique(V(V>0)))
                V2(V==r) = A_diag(r);
            end
            niftiwrite(V2,['data/nodes_edges/',models{m},'_state',num2str(i),'.nii'],info)
            
            BrainNet_MapCfg('BrainMesh_ICBM152.nv',...
                ['data/nodes_edges/',models{m},'_state',num2str(i),'.nii'],...
                'data/nodes_edges/vieweroptions_surf.mat',...
                ['data/nodes_edges/',models{m},'_state_surf',num2str(i),'.jpg'])
        end
        
        
        figure('Position',[0,0,400,400])
        tiledlayout(1,1,'Padding','none')
        if m==3
            cmap=[-0.15,0.45];
            imagesc(A,cmap);colormap jet%,colorbar
        elseif m==1
            cmap = [-0.1,1];
            imagesc(A/max(A(:)),cmap);colormap jet
        end
        xticks([])
        yticks([])
        %         xlabel('Brain area')
        axis square
        %     cb = colorbar('XTickLabel',{'-0.04','-0.01','0.01','0.04'}, ...
        %                'XTick', [-0.04,-0.01,0.01,0.04],...
        %                'FontSize',13)
        
        exportgraphics(gcf,['data/nodes_edges/',models{m},'_state_map',num2str(i),'.png'],'resolution','300');
        
    end
    figure('Position',[0,0,400,400])
    tiledlayout(1,1,'Padding','none')
    cmap=[-0.15,0.45];
    imagesc(A,cmap);colormap jet,colorbar
    xticks([])
    yticks([])
    print(gcf,['data/nodes_edges/',models{m},'_state_map_colorbar'],'-dpng','-r300');
    
end



% need to manually edit node first line to #aal90
% close all
% cd('/data1/anderssolsen/np2/nodes_edges/')
% addpath(genpath('/data1/anderssolsen/np2/BrainNetViewer'))
% BrainNet

%%

%%%%%% Same for NMI
figure
hold on
K=4;
testNMIs = nan(numel(K),5);
task_raw = table2array(readtable('data/raw/motor_ref.txt'));
task = repmat(task_raw,29,1);
task2 = repmat(task_raw(2:end,:),29,1);
addpath('src/visualization')
for m = 1:numel(models)
    testNMIs = nan(1,5);
    d = dir(['data/real_fit/K',num2str(K),models{m},'_assignment*.csv']);
    for r = 1:numel(d)
        data = table2array(readtable([d(r).folder,'/',d(r).name]));
        
        if m==1 || m==3
            testNMIs(r) = calcNMI(data',task');
        else
            data2 = data(1:end-1,:);
            for kk = 1:K
                data3(kk,:) = data2(:)==kk-1;
            end
            testNMIs(r) = calcNMI(data3,task2');
        end
    end
    
    
    errorbar(K,nanmean(testNMIs,2),nanstd(testNMIs,[],2),'color',cols{m},'DisplayName',modelnames{m},'LineWidth',1.5)
end
legend
xlim([1.5,10.5])
xlabel('Model order (K)')
ylabel('Normalized mutual information')
return

K=10
figure,subplot(2,1,1)
plot(data3'+1.1*(1:K)),xlim([0,1000])
subplot(2,1,2)
plot(task+[0,1.1]),xlim([0,1000])

figure,subplot(2,1,1)
plot(data+1.1*(1:K)),xlim([0,1000])
subplot(2,1,2)
plot(task+[0,1.1]),xlim([0,1000])






















