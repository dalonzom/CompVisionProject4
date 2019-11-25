
clear;
clc;
%% Training 
%% Read in training images and use Harris corner detection 
first = 1; 
last = 550;
featureLength = 12; 
images = imread(strcat('CarTrainImages/train_car', sprintf('%03d',first),'.jpg'));
count = 1;
harris ={}; 
for i = first:last
    count = count + 1;
    images(:,:,:,count) = imread(strcat('CarTrainImages/train_car', sprintf('%03d',i),'.jpg'));
    harris{i} = {harrisDetector(images(:,:,:,count), 1e20)}; 
end

%% Extract 19x19 image patch for each feature 
features = getPatches(harris, images, featureLength);  

%% Use Kmeans to cluster the data 
n = size(features,2);
kmean = struct();

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

% cidx = kmean(clusters).cidx; 
% ctrs = kmean(clusters).ctrs; 
% sumd = kmean(clusters).sumd; 

clusters = 30;

patches = zeros(size(features,2),size(features(1).pixels,1));
for i = 1:size(features,2)
    patches(i,:) = features(i).pixels';
end
[idx, C] = kmeans(patches, clusters, 'MaxIter', 1000);


%% Assign local patches to words in vocabulary, record possible displacement 
%% vectors between word and object center 
rowOffset = 20; 
colOffset = 50; 
vocab = buildVocab(features, idx, clusters, C, rowOffset, colOffset); 

%% Testing 
count = 7
image = imread(strcat('CarTestImages/test_car', sprintf('%03d',count),'.jpg'));
harris = {harrisDetector(image, 1e11)}; 
testFeatures = getPatches(harris,image, featureLength);
[~,idx_test] = pdist2(C,[testFeatures.pixels]','euclidean','Smallest',1);

votes = zeros(size(image));

for i = 1:size(testFeatures,2)
    location = testFeatures(i).location;
    offsets = vocab(idx_test(i)).displacments;
    for j = 1:size(offsets,1)
        yLoc = location(1) - offsets(j,1);
        xLoc = location(2) - offsets(j,2);
        if xLoc >= 1 && xLoc <= size(image,2) && yLoc >= 1 && yLoc <= size(image,1)
            votes(yLoc, xLoc) = votes(yLoc, xLoc) + 1;
        end
    end
end
filter = zeros(25,25); 
filter(13,13) = 1; 
filter = imgaussfilt(filter, 4); 
votes = imfilter(votes,filter);
%%
imageDisp = votes./max(votes(:));
figure(1)
imshow(image)
figure(2)
imshow(imageDisp)

%%
% % find the bin index for every data point
% binIndex = 1:clusters;
% equals = bsxfun(@eq,idx_test',binIndex);
% %votes = sum(equals);
% 
% % Thresholding 
% threshold = 4; 
% possibleClusters = find(votes >= threshold); 
% for i = 1:size(possibleClusters,2)
%     displacements = vocab(possibleClusters(i)).displacments; 
% end 

