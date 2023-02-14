file = readtable('data/external/Schaefer200_yeo7_lut.txt');;

labcell = file.Var5;

for i = 1:200
    labels{i} = labcell{i}(11:end);
end
save('data/external/labels_cell','labels')
% writetable(array2table(lab2'),'data/external/labels.txt','WriteVariableNames',false)