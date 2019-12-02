figure(65)
displayTarget = 42;
image = imread(strcat('CarTestImages/test_car', sprintf('%03d',displayTarget),'.jpg'));
imshow(image);
hold on;
locations = results(displayTarget).locations;
for i = 1:size(locations,1)
    loc = locations(i,:);
    x = loc(1);
    y = loc(2);
    plot([x-50 x+50], [y-20 y-20], 'y', 'LineWidth', 2)
    plot([x+50 x+50], [y-20 y+20], 'y', 'LineWidth', 2)
    plot([x+50 x-50], [y+20 y+20], 'y', 'LineWidth', 2)
    plot([x-50 x-50], [y+20 y-20], 'y', 'LineWidth', 2)
end