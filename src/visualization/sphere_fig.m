clear,close all
% cd(fileparts(which(mfilename)));
ff = 'reports/methods/'; %figure folder
%% specify sig
%sig1 = diag([1,1e-3,1e-3]);
sig2 = 1*(eye(3)+0.8*(ones(3)-eye(3)));
sig3 = diag([1e-2,1,1])+0.91*[0,0,0;0,0,1;0,1,0];
SIGMAs = cat(3,sig2,sig3);
% PI = [0.33,0.33,0.34];

%% calculate PI and transition matrix
task_raw = table2array(readtable('data/raw/motor_ref.txt'));
task = resample(task_raw(1:97,:),12,1);
PI = (task-min(task))./sum(task-min(task),2);
[~,maxseq] = max(PI,[],2);

% for all maxseq=1, which is the second one?
next1=maxseq((find(maxseq(1:end-1)==1)+1));
next2=maxseq((find(maxseq(1:end-1)==2)+1));
T(1,1) = mean(next1==1);
T(1,2) = mean(next1==2);
T(2,1) = mean(next2==1);
T(2,2) = mean(next2==2);

rng default
if ~exist('data/synthetic/HMMdata.h5')
    [X,cluster_id] = syntheticMixture3D(PI,SIGMAs,size(PI,1),0.01);
% % % %     [X,cluster_id] = syntheticHMM(PI,SIGMAs, T,length(maxseq),1);
    h5create('data/synthetic/HMMdata.h5','/X',size(X))
    h5write('data/synthetic/HMMdata.h5','/X',X)
    h5create('data/synthetic/HMMdata.h5','/cluster_id',size(cluster_id))
    h5write('data/synthetic/HMMdata.h5','/cluster_id',cluster_id)
else
    X = h5read('data/synthetic/HMMdata.h5','/X');
    cluster_id = h5read('data/synthetic/HMMdata.h5','/cluster_id');
end
%% Figure 1
pointsspherefig(X,cluster_id);
% pause(2)
% exportgraphics(gcf,[ff,'sphere_WMM_data.png'],'Resolution',300)


% WMM_results = WMM_EM_BigMem2(X,2,200,1,'++',0)




%% Figure 2
%%% Watson MM
% WMM_results = WMM_EM_BigMem2(X,2,200,1,'++',0);mu1 = WMM_results.mu(:,1);mu2 = WMM_results.mu(:,2);
mu1 = table2array(readtable('data/synthetic/Watson_MM_comp0mu.csv'));
kappa1 = table2array(readtable('data/synthetic/Watson_MM_comp0kappa.csv'));
mu2 = table2array(readtable('data/synthetic/Watson_MM_comp1mu.csv'));
kappa2 = table2array(readtable('data/synthetic/Watson_MM_comp1kappa.csv'));
orderwmm = contourspherefig([mu1,mu2],[kappa1,kappa2],[],SIGMAs);
% title('Watson mixture')
% pause(2)
% exportgraphics(gcf,[ff,'sphere_WMM_contour.png'],'Resolution',300)

%%% ACG MM
L1 = table2array(readtable('data/synthetic/ACG_MM_comp0.csv'));
L2 = table2array(readtable('data/synthetic/ACG_MM_comp1.csv'));
orderacgmm = contourspherefig([],[],cat(3,L1,L2),SIGMAs);
% gcf,title('ACG mixture')
% pause(2)
% exportgraphics(gcf,[ff,'sphere_ACGMM_contour.png'],'Resolution',300)

%%% Watson HMM
mu1 = table2array(readtable('data/synthetic/Watson_HMM_comp0mu.csv'));
kappa1 = table2array(readtable('data/synthetic/Watson_HMM_comp0kappa.csv'));
mu2 = table2array(readtable('data/synthetic/Watson_HMM_comp1mu.csv'));
kappa2 = table2array(readtable('data/synthetic/Watson_HMM_comp1kappa.csv'));
orderwmmhmm = contourspherefig([mu1,mu2],[kappa1,kappa2],[],SIGMAs);
title('Watson HMM')
% pause(2)
% exportgraphics(gcf,[ff,'sphere_WMM_contour.png'],'Resolution',300)

%%% ACG HMM
L1 = table2array(readtable('data/synthetic/ACG_HMM_comp0.csv'));
L2 = table2array(readtable('data/synthetic/ACG_HMM_comp1.csv'));
orderacgmmhmm = contourspherefig([],[],cat(3,L1,L2),SIGMAs);
% gcf,title('ACG HMM')
% pause(2)
% exportgraphics(gcf,[ff,'sphere_WMM_contour.png'],'Resolution',300)

%% emission probs
t_fMRI = linspace(0,4.1,size(X,1));

figure('Position',[50,50,400,700])
tiledlayout(10,1)
cols{1} = [0,0.5,0.5];
cols{2} = [0.5,0,0.5];

nexttile(1,[2,1]),hold on
x_wmm = table2array(readtable('data/synthetic/Watson_MM_posterior.csv'));
plot(t_fMRI,x_wmm(:,orderwmm(1))+0.2,'color',cols{1});
plot(t_fMRI,x_wmm(:,orderwmm(2))+1.4,'color',cols{2});
xticks([]),yticks([]),xlim([-0.1,4.1]),ylim([0,2.6])
ylabel('Watson')
title('Mixture model posterior probabilities')
% yticks([0.65,1.75])
% yticklabels({'\beta_1','\beta_2'})
nexttile(3,[2,1]),hold on
x_acgmm = table2array(readtable('data/synthetic/ACG_MM_posterior.csv'));
plot(t_fMRI,x_acgmm(:,orderacgmm(1))+0.2,'color',cols{1});
plot(t_fMRI,x_acgmm(:,orderacgmm(2))+1.4,'color',cols{2});
xticks([]),yticks([]),xlim([-0.1,4.1]),ylim([0,2.6])
ylabel('ACG')

nexttile(5,[2,1]),hold on
x = table2array(readtable('data/synthetic/Watson_HMM_viterbi.csv'));
x_wmmhmm = [x,x];x_wmmhmm(:,1)=x==0;x_wmmhmm(:,2)=x==1;
plot(t_fMRI,x_wmmhmm(:,orderwmmhmm(1))+0.2,'color',cols{1});
plot(t_fMRI,x_wmmhmm(:,orderwmmhmm(2))+1.4,'color',cols{2});
xticks([]),yticks([]),xlim([-0.1,4.1]),ylim([0,2.6])
ylabel('Watson')
title('Hidden Markov model state sequence')
% yticks([0.65,1.75])
% yticklabels({'\beta_1','\beta_2'})
nexttile(7,[2,1]),hold on
x = table2array(readtable('data/synthetic/ACG_HMM_viterbi.csv'));
x_acgmmhmm = [x,x];x_acgmmhmm(:,1)=x==0;x_acgmmhmm(:,2)=x==1;
plot(t_fMRI,x_acgmmhmm(:,orderacgmmhmm(1))+0.2,'color',cols{1});
plot(t_fMRI,x_acgmmhmm(:,orderacgmmhmm(2))+1.4,'color',cols{2});
xticks([]),yticks([]),xlim([-0.1,4.1]),ylim([0,2.6])
ylabel('ACG')

nexttile(9,[2,1])
plot(t_fMRI,task(:,1),'LineWidth',1.5,'color',[0,0.5,0]),hold on
plot(t_fMRI,task(:,2)+0.2+max(task(:)),'LineWidth',1.5,'color',[0.5,0,0])
title('Right/left hand motor task')
xlabel('Time [min]'),
ylim([-.3,2.7])
xlim([-.1 4.1])
xticks(0:4)
yticks([mean(task(:,1)),mean(task(:))+max(task(:))+0.2])
yticklabels({'RH','LH'})
exportgraphics(gcf,[ff,'emission_viterbi.png'],'Resolution',300)
%% function
function [X,cluster_allocation] = syntheticMixture3D(PI,SIGMAs,num_points,noise)

num_clusters = size(SIGMAs,3);
point_dim = size(SIGMAs,2);
% Lower_chol = zeros(point_dim,point_dim,num_clusters);
% for idx = 1:num_clusters
%     Lower_chol(:,:,idx) = chol(SIGMAs(:,:,idx),'lower');
% end

X = zeros(num_points,point_dim);
cluster_allocation = zeros(num_points,1);
for n = 1:num_points
    n_clust_id = randsample(num_clusters,1,true,PI(n,:));
    nx = SIGMAs(:,:,n_clust_id) * randn(point_dim,1)+noise*randn(point_dim,1);
    X(n,:) = normc(nx);
    cluster_allocation(n) = n_clust_id;
end
end

function [X_emission,Z_state_seq] = syntheticHMM(pi,SIGMAs, transition_matrix,seq_len,num_subject)

num_states = size(SIGMAs,3);
point_dim = size(SIGMAs,2);
Lower_chol = zeros(point_dim,point_dim,num_states);

for idx = 1:num_states
    Lower_chol(:,:,idx) = chol(SIGMAs(:,:,idx),'lower');
end

X_emission = zeros(num_subject, seq_len, point_dim);
Z_state_seq = zeros(num_subject, seq_len);
T_matrix = normr(transition_matrix);

for sub = 1:num_subject
    for t = 1:seq_len
        if t==1
            t_state_id = randsample(1:num_states,1,true,pi(t,:));
        else
            %             get transition probs from state at time t-1 to all states at time t
            t_z_probs = T_matrix(Z_state_seq(sub,t-1),:);
            % get state for time t by weighting the transition probs
            t_state_id = randsample(1:num_states,1,true,t_z_probs);
        end
        % Emission
        t_x = Lower_chol(:,:,t_state_id)*randn(point_dim,1);
        X_emission(sub,t,:) = normc(t_x);
        Z_state_seq(sub,t,:) = t_state_id;
    end
end

X_emission = squeeze(X_emission);
end

%% Figure with random data

function pointsspherefig(X,cluster_id)
gridPoints = 1000;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0 0 .5 1]); clf;%'visible','off',

sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(-29,-13)

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .2,'FaceColor','none','EdgeColor',[0,0,0]);
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off
view(-29,-13)

cols = [0,0.4,0;0.5,0,0;0,0,0.5];

for i = 1:numel(unique(cluster_id))
    scatter3(X(cluster_id==i,1), X(cluster_id==i,2), X(cluster_id==i,3),7,cols(i,:),'filled');
end
end

%% Figure fit, Sphere, contour
function order = contourspherefig(mu,kappa,L,target)

gridPoints = 1000;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0.5 0 .5 1]); clf;%'visible','off',
ax1 = axes;
sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
view(-29,-13)

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .2,'FaceColor','none','EdgeColor',[0,0,0]);
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off
view(-29,-13)

addpath('/dtu-compute/HCP_dFC/2023/hcp_dfc/src/models/')
%
% mu1 = WMM_results.mu(:,1);mu2 = WMM_results.mu(:,2);
% kappa = WMM_results.kappa;

% m = [m1;m2];
% [~,mini] = min(kappa)
% multi = [1,1];multi(mini)=8;
% kappa = multi'.*kappa;

T1 = nan(size(XX));T2 = nan(size(XX));

varfactor = 0.5;
likelihood_threshold = [0,-1];


if ~isempty(mu)&&~isempty(kappa)
    M2 = kummer_log(0.5,1.5,kappa,50000);
    Cp = gammaln(1.5)-log(2)-1.5*log(pi)-M2';
    %     kappa = kappa*10;
    for i = 1:size(XX,1)
        for j = 1:size(XX,2)
            tmp = [XX(i,j),YY(i,j),ZZ(i,j)];
            logpdf = Cp + kappa'.*((mu'*tmp').^2);
%             if (tmp*mu(:,1)).^2>1-varfactor./kappa(1)
%                 T1(i,j) = (tmp*mu(:,1)).^2;
%             elseif (tmp*mu(:,2)).^2>1-varfactor./kappa(2)
%                 T2(i,j) = (tmp*mu(:,2)).^2;
%             end
            if logpdf(1)>likelihood_threshold(1)
                T1(i,j) = logpdf(1);
            elseif logpdf(2)>likelihood_threshold(1)
                T2(i,j) = logpdf(2);
            end
        end
    end
elseif ~isempty(L)
    Cp = gammaln(1.5)-log(2)-1.5*log(pi);
    logdeta1 = -2*sum(log(abs(diag(L(:,:,1)))));
    logdeta2 = -2*sum(log(abs(diag(L(:,:,2)))));
    
    for i = 1:size(XX,1)
        for j = 1:size(XX,2)
            tmp = [XX(i,j),YY(i,j),ZZ(i,j)];
            %             ll1(i,j) = Cp(1)+(-1.5)*log(tmp*A(:,:,1)*tmp');
            %             ll2(i,j) = Cp(2)+(-1.5)*log(tmp*A(:,:,2)*tmp');
            
            B1 = tmp*L(:,:,1);B1 = sum(B1.*B1,2);
            B2 = tmp*L(:,:,2);B2 = sum(B2.*B2,2);
            
            if Cp-0.5*logdeta1+(-1.5)*log(B1)>likelihood_threshold(2)
                T1(i,j) = Cp-0.5*logdeta1+(-1.5)*log(B1);
            elseif Cp-0.5*logdeta2+(-1.5)*log(B2)>likelihood_threshold(2)
                T2(i,j) = Cp-0.5*logdeta2+(-1.5)*log(B2);
            end
            
            
            %             if norm(L_chol1*tmp').^2>1-varfactor
            %                 T1(i,j) = norm((L_chol1*tmp')).^2;
            %             elseif norm(L_chol2*tmp').^2>1-varfactor
            %                 T2(i,j) = norm((L_chol2*tmp')).^2;
            %             end
        end
    end
end

% d1 = m1./norm(m1);d2 = m2./norm(m2);d3 = m3./norm(m3);
% line([0,d1(1)],[0,d1(2)],[0,d1(3)],'Color',col1,'LineWidth',2.5)
% line([0,d2(1)],[0,d2(2)],[0,d2(3)],'Color',col2,'LineWidth',2.5)
% line([0,d3(1)],[0,d3(2)],[0,d3(3)],'Color',col3,'LineWidth',2.5)

ax2 = axes;
sh(3) = surf(ax2,XX,YY,ZZ);
set(sh(3),'EdgeColor','none');
set(sh(3),'CData',(T1-min(T1(:)))./(max(T1(:))-min(T1(:))));
view(-29,-13)

ax3 = axes;
sh(4) = surf(ax3,XX,YY,ZZ);
set(sh(4),'EdgeColor','none');
set(sh(4),'CData',(T2-min(T2(:)))./(max(T2(:))-min(T2(:))));
view(-29,-13)

% ax4 = axes;
% sh(4) = surf(ax4,XX,YY,ZZ);
% set(sh(4),'EdgeColor','none');
% set(sh(4),'CData',(T3-min(T3(:)))./(max(T3(:))-min(T3(:))));


hlink = linkprop([ax1,ax2,ax3],{'XLim','YLim','ZLim','CameraUpVector','CameraPosition','CameraTarget','CameraViewAngle'});
% linkaxes([ax1,ax2,ax3,ax4])
ax2.Visible = 'off';
ax2.XTick = [];
ax2.YTick = [];
ax3.Visible = 'off';
ax3.XTick = [];
ax3.YTick = [];
% ax4.Visible = 'off';
% ax4.XTick = [];
% ax4.YTick = [];

col1 = [0,0.4,0];
col2 = [0.5,0,0];
% col3 = [0.5,0,0.5];

% cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0,256)']);
% cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0,256)']);

cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0.5,256)']);
cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0.5,256)']);

%

if ~isempty(mu)&&~isempty(kappa)
    if ((target(:,:,1)*ones(3,1))'*mu(:,1)).^2<((target(:,:,1)*ones(3,1))'*mu(:,2)).^2
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
        order = [2,1];
    else
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
        order = [1,2];
    end
elseif ~isempty(L)
    if norm(inv(L(:,:,1)*L(:,:,1)')./norm(inv(L(:,:,1)*L(:,:,1)'))-target(:,:,1)./norm(target(:,:,1)))>norm(inv(L(:,:,2)*L(:,:,2)')./norm(inv(L(:,:,2)*L(:,:,2)'))-target(:,:,1)./norm(target(:,:,1)))
        colormap(ax2,cmaps{1})
        colormap(ax3,cmaps{2})
        order = [2,1];
    else
        colormap(ax2,cmaps{2})
        colormap(ax3,cmaps{1})
        order = [1,2];
    end
end




end










