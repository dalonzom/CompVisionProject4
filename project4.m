% 
% clear;
% clc;
% %% Training 
% %% Read in training images and use Harris corner detection 
% first = 1; 
% last = 550; 
% images = imread(strcat('CarTrainImages/train_car', sprintf('%03d',first),'.jpg'));
% count = 1;
% harris ={}; 
% for i = first:last
%     count = count + 1;
%     images(:,:,:,count) = imread(strcat('CarTrainImages/train_car', sprintf('%03d',i),'.jpg'));
%     harris{i} = {harrisDetector(images(:,:,:,count), 100)}; 
% end
% 
% %% Extract 19x19 image patch for each feature 
% features = getPatches(harris, images, 9);  
% 
% %% Use Kmeans to cluster the data 
% n = size(features,2);
% kmean = struct();
% 
% % for i = 1:n/10 
% %     [cidx, ctrs, sumd] = kmeans([double(cell2mat(features))]', i, 'MaxIter',1000);
% %     kmean(i).cidx = cidx; 
% %     kmean(i).ctrs = ctrs; 
% %     kmean(i).sumd = sumd; 
% % %     prior = i -1; 
% % %     means = i*size(features,1); 
% % %     covar = i*size(features,1) *(size(features,1) +1)/2; 
% % %     pk = prior+means+covar;
% % %     pk = 2; 
% % %     kmean(i).bic = sum(sumd) - pk*log(i); 
% %     
% %     kmean(i).bic =n*log(sum(sumd)/n)+(i*3)*log(n); 
% %     
% % end 
% 
% % cidx = kmean(clusters).cidx; 
% % ctrs = kmean(clusters).ctrs; 
% % sumd = kmean(clusters).sumd; 
% 
% clusters = 1000;
% 
% patches = zeros(size(features,2),size(features(1).pixels,1));
% for i = 1:size(features,2)
%     patches(i,:) = features(i).pixels';
% end
% [idx, C] = kmeans(patches,clusters);
% 
% 
% %% Assign local patches to words in vocabulary, record possible displacement 
% %% vectors between word and object center 
% vocab = buildVocab(features, idx, clusters, C, 20, 50); 

%% Testing 
count = 1; 
image = imread(strcat('CarTestImages/test_car', sprintf('%03d',count),'.jpg'));
harris = {harrisDetector(image, 100)}; 
features = getPatches(harris, image, 9);
[~,idx_test] = pdist2(C,[features.pixels]','euclidean','Smallest',1);

