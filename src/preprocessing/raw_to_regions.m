clear

datadir = dir('data/raw/**/*.nii.gz')

atlas = int16(niftiread('data/external/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_3mm.nii'));
atlas = reshape(atlas,prod(size(atlas,[1,2,3])),1);


for i = 1:numel(datadir)
    V = niftiread([datadir(i).folder,'/',datadir(i).name]);
    V = reshape(V,prod(size(V,[1,2,3])),size(V,4));
    
    for region = 1:max(atlas)
        
        V_ROI(:,region) = mean(V(atlas==region,:),1);
        
    end
    
    delete(['data/processed/atlas_',datadir(i).name(1:4),'.h5'])
    h5create(['data/processed/atlas_',datadir(i).name(1:4),'.h5'],'/data',size(V_ROI))
    h5write(['data/processed/atlas_',datadir(i).name(1:4),'.h5'],'/data',V_ROI)
    
    
    disp(['Done with ',num2str(i)])
end