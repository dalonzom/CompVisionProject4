figure(45)
for i=1:size(features,2)
    plot(features(i).location(2),features(i).location(1),'*')
    hold on
end