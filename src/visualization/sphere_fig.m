clear,close all
% cd(fileparts(which(mfilename)));
ff = 'reports/methods/'; %figure folder
%% specify sig
%sig1 = diag([1,1e-3,1e-3]);
sig2 = eye(3)+0.99*(ones(3)-eye(3));
sig3 = diag([1e-2,1,1])+0.9*[0,0,0;0,0,1;0,1,0];
SIGMAs = cat(3,sig2,sig3);
% PI = [0.33,0.33,0.34];

%% calculate PI and transition matrix
task = resample(table2array(readtable('data/raw/motor_ref.txt')),12,1);
PI = (task-min(task))./sum(task-min(task),2);
[~,maxseq] = max(PI,[],2);

% for all maxseq=1, which is the second one?
next1=maxseq((find(maxseq(1:end-1)==1)+1));
next2=maxseq((find(maxseq(1:end-1)==2)+1));
T(1,1) = mean(next1==1);
T(1,2) = mean(next1==2);
T(2,1) = mean(next2==1);
T(2,2) = mean(next2==2);

% [X, cluster_id] = syntheticMixture3D(PI, SIGMAs, size(PI,1));
rng default
[X,cluster_id] = syntheticHMM(PI,SIGMAs, T,size(PI,1),1);

%% Figure random data

gridPoints = 1000;
[x,y,z] = sphere(gridPoints);
figure('units','normalized','outerposition',[0 0 .5 1]); clf;%'visible','off',

sh(1) = surf(x,y,z, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .2,'FaceColor','none','EdgeColor',[0,0,0]);

%
% figure;
cols = [0,0.4,0;0.5,0,0;0,0,0.5];
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off

for i = 1:size(SIGMAs,3)
scatter3(X(cluster_id==i,1), X(cluster_id==i,2), X(cluster_id==i,3),7,cols(i,:),'filled');
end

view(-29,-13)
exportgraphics(gcf,[ff,'sphere_WMM_data.png'],'Resolution',300)
%% Figure fit, Sphere, contour

gridPoints = 1000;
[XX,YY,ZZ] = sphere(gridPoints);
figure('units','normalized','outerposition',[0.5 0 .5 1]); clf;%'visible','off',
ax1 = axes;
sh(1) = surf(XX,YY,ZZ, 'FaceAlpha', .2, 'EdgeAlpha', .1,'EdgeColor','none','FaceColor','none');
hold on; axis equal;
xlabel('x'); ylabel('y'); zlabel('z');

% smaller sphere to show lines on
[x2,y2,z2] = sphere(20); %30
sh(2) = surf(x2,y2,z2, 'EdgeAlpha', .2,'FaceColor','none','EdgeColor',[0,0,0]);
set(gca,'XColor', 'none','YColor','none','ZColor','none')
grid off

addpath('/dtu-compute/HCP_dFC/2023/hcp_dfc/src/models/')
WMM_results = WMM_EM_BigMem2(X,2,200,1,'++',0)

mu1 = WMM_results.mu(:,1);mu2 = WMM_results.mu(:,2);
% m = [m1;m2];
[~,mini] = min(WMM_results.kappa)
multi = [1,1];multi(mini)=8;
kappa = multi'.*WMM_results.kappa;

T1 = nan(size(XX));T2 = nan(size(XX));

varfactor = 4;

for i = 1:size(XX,1)
    for j = 1:size(XX,2)
        tmp = [XX(i,j),YY(i,j),ZZ(i,j)];
        if (tmp*mu1).^2>1-varfactor./kappa(1)
            T1(i,j) = (tmp*mu1).^2;
        elseif (tmp*mu2).^2>1-varfactor./kappa(2)
            T2(i,j) = (tmp*mu2).^2;
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

ax3 = axes;
sh(4) = surf(ax3,XX,YY,ZZ);
set(sh(4),'EdgeColor','none');
set(sh(4),'CData',(T2-min(T2(:)))./(max(T2(:))-min(T2(:))));

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

cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0,256)']);
cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0,256)']);

% 
colormap(ax2,cmaps{1})
colormap(ax3,cmaps{2})
view(-29,-13)
exportgraphics(gcf,[ff,'sphere_WMM_contour.png'],'Resolution',300)
close all
%% function
function [X,cluster_allocation] = syntheticMixture3D(PI,SIGMAs,num_points)

num_clusters = size(SIGMAs,3);
point_dim = size(SIGMAs,2);
Lower_chol = zeros(point_dim,point_dim,num_clusters);
for idx = 1:num_clusters
    Lower_chol(:,:,idx) = chol(SIGMAs(:,:,idx),'lower');
end

X = zeros(num_points,point_dim);
cluster_allocation = zeros(num_points);
for n = 1:num_points
    n_clust_id = randsample(num_clusters,1,true,PI(n,:));
    nx = Lower_chol(:,:,n_clust_id) * randn(point_dim,1);
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










