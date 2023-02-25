%%
clear
addpath(genpath('BrainNetViewer'))

% states = dir('data/real_K/K5ACG_MM_comp*0.csv')
states = dir('data/nodes_edges/K3ACG_MM_comp*0.csv');

for i = 1:numel(states)
    
    A = table2array(readtable([states(i).folder,'/',states(i).name]));
    A_diag = diag(A);
    %     A = A./max(abs(A(:)));
%     M = diag(A);
%     Mpos = 
    
    nodes = readtable('Node_Schaefer100.node','FileType','text');
    nodes.Var5 = A_diag./max(A_diag);
%     nodes.Var4(Mpos) = 2;
%     nodes.Var4(Mneg) = 1;
    nodes.Properties.VariableNames = {'# Schaefer_100','2','3','4','5','6'};
    writetable(nodes,['data/nodes_edges/nodes_state',num2str(i),'.node'],'FileType','text','Delimiter','\t')
    
    % if threshold at 75th quantile edge strength
    A2 = A;
    A2(A2==diag(diag(A2))) = 0;
    A2 = A2./max(abs(A2(:)));
    A2(abs(A2(:))<0.9) = 0;
%     A2(abs(A2(:))<quantile(abs(A2(:)),0.75)) = 0;
    edges = array2table(A2);
    writetable(edges,['data/nodes_edges/edges_state',num2str(i),'.edge'],'FileType','text','Delimiter','\t','WriteVariableNames',false)
    
    figure,
    subplot(1,2,1),imagesc(A),colorbar,axis square
    subplot(1,2,2),imagesc(A2),colorbar,axis square
    
    % remember to manually edit the node file line 1 to #aal90
    
    BrainNet_MapCfg('BrainMesh_ICBM152.nv',...
        ['data/nodes_edges/nodes_state',num2str(i),'.node'],...
        ['data/nodes_edges/edges_state',num2str(i),'.edge'],...
        'data/nodes_edges/options.mat',...
        ['data/nodes_edges/ACGMM_state_edge',num2str(i),'.jpg'])
    
    V = niftiread('data/external/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii');
    info = niftiinfo('data/external/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_3mm.nii');
    info.Datatype = 'double';
    V2 = zeros(size(V));
    A_diag = A_diag./norm(A_diag);
    for r = 1:numel(unique(V(V>0)))
        V2(V==r) = A_diag(r);
    end
    niftiwrite(V2,['data/nodes_edges/ACGMM_state',num2str(i),'.nii'],info)
    
    BrainNet_MapCfg('BrainMesh_ICBM152.nv',...
        ['data/nodes_edges/ACGMM_state',num2str(i),'.nii'],...
        'data/nodes_edges/vieweroptions_surf.mat',...
        ['data/nodes_edges/ACGMM_state_surf',num2str(i),'.jpg'])
    
    figure('Position',[0,0,200,200])
    tiledlayout(1,1,'Padding','none')
    imagesc(A);colormap jet,colorbar
    xlabel('Brain area')
    axis square
    %     cb = colorbar('XTickLabel',{'-0.04','-0.01','0.01','0.04'}, ...
    %                'XTick', [-0.04,-0.01,0.01,0.04],...
    %                'FontSize',13)
    
    print(gcf,['data/nodes_edges/ACGMM_state_map',num2str(i)],'-dpng','-r300');
    
end
% need to manually edit node first line to #aal90
close all
cd('/data1/anderssolsen/np2/nodes_edges/')
% addpath(genpath('/data1/anderssolsen/np2/BrainNetViewer'))
% BrainNet
