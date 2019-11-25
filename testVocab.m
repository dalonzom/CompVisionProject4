figure(55)
for i=1:size(vocab,2)
    plot(vocab(i).displacments(:,2),vocab(i).displacments(:,1),'*')
    hold on
end