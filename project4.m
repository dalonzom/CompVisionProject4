
% clear;
% clc;
%% Read in training images and use Harris corner detection 
first = 1; 
last = 550; 
images = imread(strcat('CarTrainImages/train_car', sprintf('%03d',first),'.jpg'));
count = 1;
harris ={}; 
for i = first:last
    count = count + 1;
    images(:,:,:,count) = imread(strcat('CarTrainImages/train_car', sprintf('%03d',i),'.jpg'));
    harris{i} = {harrisDetector(images(:,:,:,count), 100)}; 
end

%% Extract 9x9 image patch for each feature 
features = struct();
count = 1; 
for i = first:last
    mat = cell2mat(harris{i});
    image = images(:,:,:,i); 
    [rows, columns, ~] = find(mat ~= 0); 
    for j = 1:size(rows,1)
        row = rows(j); 
        col = columns(j);
        if row > 4 && col > 4 && row < 36 && col < 96 
            vec = image(row-4:row+4, col-4:col+4);
            features(count).pixels = reshape(vec,81,1); 
            features(count).location = [row, col]; 
            count = count + 1; 
        end
    end 
    
end 

%% Use Kmeans to cluster the data 
n = size(features,2);
% kmean = struct();
% 
% for i = 1:n/10 
%     [cidx, ctrs, sumd] = kmeans([double(cell2mat(features))]', i, 'MaxIter',1000);
%     kmean(i).cidx = cidx; 
%     kmean(i).ctrs = ctrs; 
%     kmean(i).sumd = sumd; 
% %     prior = i -1; 
% %     means = i*size(features,1); 
% %     covar = i*size(features,1) *(size(features,1) +1)/2; 
% %     pk = prior+means+covar;
% %     pk = 2; 
% %     kmean(i).bic = sum(sumd) - pk*log(i); 
%     
%     kmean(i).bic =n*log(sum(sumd)/n)+(i*3)*log(n); 
%     
% end 
clusters = 29; 
cidx = kmean(clusters).cidx; 
ctrs = kmean(clusters).ctrs; 
sumd = kmean(clusters).sumd; 


%% Assign local patches to words in vocabulary, record possible displacement 
%% vectors between word and object center 
ssdDistances = struct(); 
imageCenter = [20, 50]; 
centerDistances = struct(); 
for i = 1:n
    clusterDistances = zeros(clusters, 1); 
    for j = 1:clusters 
        V = bsxfun(@minus, [double(features(i).pixels)], ctrs(j,:)');
        clusterDistances(j) = sqrt(sum(V'.^2, 2)); 
    end 
    [~, ind] = min(clusterDistances); 
    ssdDistances(i).coords =  ctrs(ind, :); 
    ssdDistances(i).cluster = ind; 
    
    centerDistances(i).cluster = ind; 
    centerDistances(i).distance = pdist([[features(i).location]; imageCenter]); 
end 

