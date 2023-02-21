file = readtable('data/external/Schaefer100_yeo7_lut.txt');

labcell = file.Var5;

for i = 1:numel(labcell)
    labels{i} = labcell{i}(11:end);
end
save('data/external/labels_cell_100','labels')
% writetable(array2table(lab2'),'data/external/labels.txt','WriteVariableNames',false)