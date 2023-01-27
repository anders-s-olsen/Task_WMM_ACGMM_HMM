close all
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


%% original data for methods figure

sig2 = eye(3)+0.99*(ones(3)-eye(3)); %noise is one minus the off diagonal element, log space
sig3 = diag([1e-02,1,1])+0.9*[0,0,0;0,0,1;0,1,0]; %noise is the first diagonal element, log space
SIGMAs = cat(3,sig2,sig3);
[X,cluster_id] = syntheticMixture3D(PI,SIGMAs,size(PI,1),0.000001);
pointsspherefig(X,cluster_id);
delete('data/synthetic_methods/HMMdata_orig.h5')
h5create('data/synthetic_methods/HMMdata_orig.h5','/X',size(X))
h5write('data/synthetic_methods/HMMdata_orig.h5','/X',X)
h5create('data/synthetic_methods/HMMdata_orig.h5','/cluster_id',size(cluster_id))
h5write('data/synthetic_methods/HMMdata_orig.h5','/cluster_id',cluster_id)

%% generate data according to noise levels

noise = logspace(-4,-1,9);
noisedB = 20*log10(noise);

for i = 1:numel(noise)
    sig2 = eye(3)+(1-noise(i))*(ones(3)-eye(3)); %noise is one minus the off diagonal element, log space
    sig3 = diag([noise(i),1,1])+0.9*[0,0,0;0,0,1;0,1,0]; %noise is the first diagonal element, log space
    
    SIGMAs = cat(3,sig2,sig3);
    [X,cluster_id] = syntheticMixture3D(PI,SIGMAs,size(PI,1),0.000001);
    %     pointsspherefig(X,cluster_id);
    delete(['data/synthetic_noise/HMMdata_noise_',num2str(noisedB(i)),'.h5'])
    h5create(['data/synthetic_noise/HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/X',size(X))
    h5write(['data/synthetic_noise/HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/X',X)
    h5create(['data/synthetic_noise/HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/cluster_id',size(cluster_id))
    h5write(['data/synthetic_noise/HMMdata_noise_',num2str(noisedB(i)),'.h5'],'/cluster_id',cluster_id)
    
end


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
    nx = chol(SIGMAs(:,:,n_clust_id),'lower') * randn(point_dim,1)+noise*randn(point_dim,1);
    
%     nx = SIGMAs(:,:,n_clust_id) * ones(point_dim,1)+noise*randn(point_dim,1);
    X(n,:) = normc(nx);
    cluster_allocation(n) = n_clust_id;
end
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