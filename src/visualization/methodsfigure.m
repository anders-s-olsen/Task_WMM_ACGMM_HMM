clear,close all
ff = 'reports/methods/'; %figure folder
%% load example dataset, fMRI
V = double(squeeze(niftiread('data/raw/ID01/ID01_4D_motor.nii.gz')));
atlas = int16(niftiread('data/external/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_3mm.nii'));
V = reshape(V,prod(size(V,[1,2,3])),size(V,4));
atlas = reshape(atlas,prod(size(atlas,[1,2,3])),1);

%% filter data

for i = 1:numel(unique(atlas(atlas>0)))
    V_atlas(:,i) = mean(V(atlas==i,:),1);
    numvox(i) = sum(atlas==i); %fine
end

% Compute eigenvectors
TR = 2.49;%s
fnq=1/(2*TR);                 % Nyquist frequency
flp = 0.009;                    % lowpass frequency of filter (Hz)
fhi = 0.08;                    % highpass
Wn=[flp/fnq fhi/fnq];         % butterworth bandpass non-dimensional frequency
k=2;                          % 2nd order butterworth filter
[bfilt,afilt]=butter(k,Wn);   % construct the filter

V_filt = filtfilt(bfilt,afilt,V_atlas);
V_filt = normc(V_filt);
V_hil = hilbert(V_filt);

V_phase = angle(V_hil);
V_abs = abs(V_hil);

V_filt_norm = normc(V_filt);

%% Simple time series plot, fMRI, 5min
t_fMRI = TR:TR:TR*size(V_atlas,1);
V_subs5 = V_filt_norm(:,1:9);
V_subs5(:,2:4) = nan;
figure('Position',[50,50,500,300])
plot(t_fMRI/60,V_subs5(1:numel(t_fMRI),:)+0.05*[-5,0,0,0,4:3:18],'k-','LineWidth',1.5)
set(gca,'box','off')
xlim([-.1 10.1])
yticks(0.05*[-5,16.5]),yticklabels({'P','1'}),ylabel('Region'),xlabel('Time [min]'),%title('fMRI time-series')
exportgraphics(gca,[ff,'methods_ts_fMRI.png'],'Resolution',300,'BackgroundColor','none')

%% Simple time series plot, Hilbert fMRI, 5min
t_fMRI = TR:TR:TR*size(V_atlas,1);
V_subs5 = V_phase(:,1:9);
V_subs5(:,2:4) = nan;
figure('Position',[50,50,500,300])
plot(t_fMRI/60,V_subs5(1:numel(t_fMRI),:)+2.2*[-5,0,0,0,4:3:18],'k-','LineWidth',1.5)
set(gca,'box','off')
xlim([-.1 10.1])
yticks(2.2*[-5,16.5]),yticklabels({'P','1'}),ylabel('Region'),xlabel('Time [min]'),%title('fMRI time-series')
exportgraphics(gca,[ff,'methods_Hilbert_fMRI.png'],'Resolution',300,'BackgroundColor','none')

% %% Hilbert figure, fMRI
% t_fMRI = TR:TR:TR*size(V_atlas,1);
% voxidx = 1000;
% 
% figure('Position',[50,50,500,275])
% plot(t_fMRI/60,1.5+normc(real(V_hil(1:numel(t_fMRI),voxidx))),'k-','LineWidth',1.5),hold on
% plot(t_fMRI/60,1.2+normc(imag(V_hil(1:numel(t_fMRI),voxidx))),'k-','LineWidth',1.5)
% plot(t_fMRI/60,0.6+3*normc(V_abs(1:numel(t_fMRI),voxidx)),'k--','LineWidth',1.5)
% plot(t_fMRI/60,0.5+normc(V_phase(1:numel(t_fMRI),voxidx)),'k-.','LineWidth',1.5)
% 
% set(gca,'box','off')
% xlim([-.1 5.1])
% ylim([0.3,1.7])
% yticks([0.5,0.75,1.2,1.5]),yticklabels({'\theta(t)','a(t)','s_h(t)','s(t)'})
% %title('Hilbert transform'),
% xlabel('Time [min]')
% % xlabel('Time [s]')
% % yticks([0,5,15]),yticklabels({'Hilbert \theta(t)','Hilbert a(t)','BOLD s(t)'})
% exportgraphics(gca,[ff,'methods_hilbertts_fMRI.png'],'Resolution',300,'BackgroundColor','none')

% %% circle with hilbert
% 
% voxidx = 1000;
% timeidx = 1:7:50;
% figure('Position',[50,50,500,500]),hold on
% phases = -pi:0.0001:pi;
% plot(cos(phases),sin(phases),'-','color',[0.5,0.5,0.5],'LineWidth',2),
% plot(0,0,'k.')
% fac = 15;
% 
% for i = 1:numel(timeidx)
%     tmp = V_hil(timeidx(i),voxidx);
%     if 15*abs(tmp)>1
%         h = quiver(0,0,fac*real(tmp),fac*imag(tmp),0,'-','color',[0.5,0.5,0.5],'LineWidth',1.5,'MaxHeadSize',.5)
%         tmp2 = tmp./abs(tmp);
%         h = quiver(0,0,real(tmp2),imag(tmp2),0,'-','color',[0,0,0],'LineWidth',1.5,'MaxHeadSize',.5)
%     elseif 15*abs(tmp)<1
%         h = quiver(0,0,fac*real(tmp),fac*imag(tmp),0,'-','color',[0.5,0.5,0.5],'LineWidth',1.5,'MaxHeadSize',.5)
%         tmp2 = tmp./abs(tmp);
%         h = quiver(real(tmp),imag(tmp),real(tmp2)-real(tmp),imag(tmp2)-imag(tmp),0,'-','color',[0,0,0],'LineWidth',1.5,'MaxHeadSize',.5)
%     end
% end
% 
% line([-1.2,1.2],[0,0],'LineWidth',2,'Color','black')
% line([0,0],[-1.2,1.2],'LineWidth',2,'Color','black')
% axis([-1.2,1.2,-1.2,1.2]),axis square
% xticks([]),yticks([]),box off
% set(gca,'Visible','off')
% exportgraphics(gca,[ff,'methods_hilbertcircle.png'],'Resolution',300,'BackgroundColor','none')
% close
%% corrplot

timeidx = [12:14];
for i = 1:numel(timeidx)
    
    cohmat = cos(V_phase(timeidx(i),:)'-V_phase(timeidx(i),:));
    
    figure('Position',[50,50,400,300]),
    imagesc(cohmat,[-1,1])
    colormap copper
    % xlabel('Brain region')
    % ylabel('Brain region')
    yticks([]),xticks([])
    if i==1
        cb = colorbar;
        set(cb,'position',[.16 .11 .03 .325])
    end
    axis square
    
    exportgraphics(gca,[ff,'cohmat_fMRI_',num2str(i),'.png'],'Resolution',300)
%     close
end
%% many leading eigenvectors

timeidx = [12:14];
for i = 1:numel(timeidx)
    cohmat = cos(V_phase(timeidx(i),1:3:end)'-V_phase(timeidx(i),1:3:end));
    
    [V1,~] = eigs(cohmat,1);
    
    ROIs = 1:numel(V1);
    pos = V1>0;neg = V1<0;
    minposwidth = min(diff(ROIs(pos)));
    if isempty(minposwidth)
        widthpos = 0.5;
    else
        widthpos = 0.5/minposwidth;
    end
    
    figure('Position',[50,50,100,300]),
    barh(ROIs(pos),V1(pos),widthpos,'r','LineWidth',0.001),hold on
    barh(ROIs(neg),V1(neg),0.5,'b','LineWidth',0.001)
    xlim([-0.2 0.2])
    yticks([])
    if i==1
        xticks([-0.1 0.1])
    else xticks([])
    end
    % ylabel('Brain region')
    % yticks(20:20:100)
    box on
    
    exportgraphics(gca,[ff,'leadeig_fMRI_',num2str(i),'.png'],'Resolution',300,'BackgroundColor','none')
    close
end

% %% Sphere, points with directions
% 
% col1 = [0,0.4,0];
% col2 = [0.5,0.5,0];
% col3 = [0.5,0,0.5];
% 
% % Generate a sphere consisting of 20by 20 faces
% hFig = figure;
% [XX,YY,ZZ]=sphere(128);
% % use surf function to plot
% s=surf(XX,YY,ZZ);  hold on
% set(s,'EdgeColor','none','FaceColor','none');hold on
% axis square
% xticks([]),yticks([]),zticks([]),
% spacing = 13;  % play around so it fits the size of your data set
% for i = 1 : spacing : length(XX(:,1))
%     plot3(XX(i,:), YY(i,:), ZZ(i,:),'-k');
%     plot3(XX(i,:), YY(i,:), ZZ(i,:),'-k');
% end
% for i = 1 : spacing : length(XX(:,1))
%     plot3(XX(:,i), YY(:,i), ZZ(:,i),'-k');
%     plot3(XX(:,i), YY(:,i), ZZ(:,i),'-k');
% end
% 
% % data from k=3
% % three main directions
% 
% numpoints = 500;
% clearvars c1 c2 c3
% m1 = [.85,0.1,-0.1];
% m2 = [-.2,.5,.2];
% m3 = [-.4,0,.6];
% for i = 1:numpoints
%     c1(i,:) = normrnd(m1,0.1);
%     c2(i,:) = normrnd(m2,0.1);
%     c3(i,:) = normrnd(m3,0.1);
% end
% c1 = c1./vecnorm(c1,2,2);
% c2 = c2./vecnorm(c2,2,2);
% c3 = c3./vecnorm(c3,2,2);
% 
% plot3(c1(:,1),c1(:,2),c1(:,3),'.','Color',col1,'MarkerSize',3)
% plot3(c2(:,1),c2(:,2),c2(:,3),'.','Color',col2,'MarkerSize',3)
% plot3(c3(:,1),c3(:,2),c3(:,3),'.','Color',col3,'MarkerSize',3)
% plot3(-c1(:,1),-c1(:,2),-c1(:,3),'.','Color',col1,'MarkerSize',3)
% plot3(-c2(:,1),-c2(:,2),-c2(:,3),'.','Color',col2,'MarkerSize',3)
% plot3(-c3(:,1),-c3(:,2),-c3(:,3),'.','Color',col3,'MarkerSize',3)
% 
% 
% d1 = m1./norm(m1);d2 = m2./norm(m2);d3 = m3./norm(m3);
% line([0,d1(1)],[0,d1(2)],[0,d1(3)],'Color',col1,'LineWidth',2.5)
% line([0,d2(1)],[0,d2(2)],[0,d2(3)],'Color',col2,'LineWidth',2.5)
% line([0,d3(1)],[0,d3(2)],[0,d3(3)],'Color',col3,'LineWidth',2.5)
% color = get(hFig,'Color');
% set(gca,'XColor',color,'YColor',color,'ZColor',color,'TickDir','out')
% view(67,16)
% exportgraphics(gca,[ff,'sphere_WMM.png'],'Resolution',300)
% 
% %% Sphere, contour
% 
% % connected with above cell
% [XX,YY,ZZ]=sphere(128);
% 
% data = [c1;-c1;c2;-c2;c3;-c3];
% results = WMM_EM_BigMem2(data,3,1000,2,'++',0)
% [~,maxidx] = max(results.ll);
% mu1 = results.mu(:,1,maxidx);mu2 = results.mu(:,2,maxidx);mu3 = results.mu(:,3,maxidx);
% m = [m1;m2;m3];
% kappa = results.kappa(:,maxidx);
% 
% T1 = nan(size(XX));T2 = nan(size(XX));T3 = nan(size(XX));
% 
% varfactor = 4;
% 
% for i = 1:size(XX,1)
%     for j = 1:size(XX,2)
%         tmp = [XX(i,j),YY(i,j),ZZ(i,j)];
%         if (tmp*mu1).^2>1-varfactor./kappa(1)
%             T1(i,j) = (tmp*mu1).^2;
%         elseif (tmp*mu2).^2>1-varfactor./kappa(2)
%             T2(i,j) = (tmp*mu2).^2;
%         elseif (tmp*mu3).^2>1-varfactor./kappa(3)
%             T3(i,j) = (tmp*mu3).^2;
%         end
%     end
% end
% 
% hFig2 = figure;
% ax1 = axes;
% sh(1)=surf(ax1,XX,YY,ZZ);  hold on
% set(sh(1),'EdgeColor','none','FaceColor','none');hold on
% axis square
% xticks([]),yticks([]),zticks([]),
% spacing = 13;  % play around so it fits the size of your data set
% for i = 1 : spacing : length(XX(:,1))
%     plot3(XX(i,:), YY(i,:), ZZ(i,:),'-k');
%     plot3(XX(i,:), YY(i,:), ZZ(i,:),'-k');
% end
% for i = 1 : spacing : length(XX(:,1))
%     plot3(XX(:,i), YY(:,i), ZZ(:,i),'-k');
%     plot3(XX(:,i), YY(:,i), ZZ(:,i),'-k');
% end
% 
% d1 = m1./norm(m1);d2 = m2./norm(m2);d3 = m3./norm(m3);
% line([0,d1(1)],[0,d1(2)],[0,d1(3)],'Color',col1,'LineWidth',2.5)
% line([0,d2(1)],[0,d2(2)],[0,d2(3)],'Color',col2,'LineWidth',2.5)
% line([0,d3(1)],[0,d3(2)],[0,d3(3)],'Color',col3,'LineWidth',2.5)
% 
% color = get(ax1,'Color');
% set(gca,'XColor',color,'YColor',color,'ZColor',color,'TickDir','out')
% 
% ax2 = axes;
% sh(2) = surf(ax2,XX,YY,ZZ);
% set(sh(2),'EdgeColor','none');
% set(sh(2),'CData',(T1-min(T1(:)))./(max(T1(:))-min(T1(:))));
% 
% ax3 = axes;
% sh(3) = surf(ax3,XX,YY,ZZ);
% set(sh(3),'EdgeColor','none');
% set(sh(3),'CData',(T2-min(T2(:)))./(max(T2(:))-min(T2(:))));
% 
% ax4 = axes;
% sh(4) = surf(ax4,XX,YY,ZZ);
% set(sh(4),'EdgeColor','none');
% set(sh(4),'CData',(T3-min(T3(:)))./(max(T3(:))-min(T3(:))));
% hlink = linkprop([ax1,ax2,ax3,ax4],{'XLim','YLim','ZLim','CameraUpVector','CameraPosition','CameraTarget','CameraViewAngle'});
% % linkaxes([ax1,ax2,ax3,ax4])
% ax2.Visible = 'off';
% ax2.XTick = [];
% ax2.YTick = [];
% ax3.Visible = 'off';
% ax3.XTick = [];
% ax3.YTick = [];
% ax4.Visible = 'off';
% ax4.XTick = [];
% ax4.YTick = [];
% 
% col1 = [0,0.4,0];
% col2 = [0.5,0.5,0];
% col3 = [0.5,0,0.5];
% 
% cmaps{1} = ([linspace(1,0,256)',linspace(1,0.5,256)',linspace(1,0,256)']);
% cmaps{2} = ([linspace(1,0.5,256)',linspace(1,0.5,256)',linspace(1,0,256)']);
% cmaps{3} = ([linspace(1,0.5,256)',linspace(1,0,256)',linspace(1,0.5,256)']);
% 
% [~,cmapidx1] = max((mu1'*m').^2);
% [~,cmapidx2] = max((mu2'*m').^2);
% [~,cmapidx3] = max((mu3'*m').^2);
% 
% colormap(ax2,cmaps{cmapidx1})
% colormap(ax3,cmaps{cmapidx2})
% colormap(ax4,cmaps{cmapidx3})
% view(67,16)
% exportgraphics(gcf,[ff,'sphere_WMM_contour.png'],'Resolution',300)
% 
% %% time series with state prob
% 
% t_fMRI = TR:TR:300;
% V_subs5 = V_filt_norm(:,1000:1000:9000);
% V_subs5(:,2:4) = nan;
% figure('Position',[50,50,500,300])
% tiledlayout(3,1)
% nexttile([1,2])
% plot(t_fMRI/60,V_subs5(1:numel(t_fMRI),:)+0.05*[-5,0,0,0,4:3:18],'k-','LineWidth',1.5)
% set(gca,'box','off')
% xlim([-.1 5.1]),xticks([])
% yticks(0.05*[-5,16.5]),yticklabels({'P','1'}),ylabel('Voxel'),%title('fMRI time-series')
% 
% nexttile(3),hold on
% % plot(t_fMRI/60,)
% 
% xlabel('Time [min]'),
% exportgraphics(gca,[ff,'methods_ts_fMRI.png'],'Resolution',300,'BackgroundColor','none')



