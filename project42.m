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
    images(:,:,count) = imread(strcat('CarTrainImages/train_car', sprintf('%03d',i),'.jpg'));
    harris{i} = {harrisDetector(images(:,:,count), 3e10)};
end

%% Extract 25x25 image patch for each feature
features = getPatches(harris, images, featureLength);

%% Use Kmeans to cluster the data
clusters = 100;
patches = zeros(size(features,2),size(features(1).pixels,1));
for i = 1:size(features,2)
    patches(i,:) = features(i).pixels';
end
[idx, C] = kmeans(patches,clusters);


%% Assign local patches to words in vocabulary, record possible displacement
%% vectors between word and object center
rowOffset = 20;
colOffset = 50;
vocab = buildVocab(features, idx, clusters, C, rowOffset, colOffset);

%% Testing
load('GroundTruth/CarsGroundTruthBoundingBoxes.mat')
results = struct();
for count = 1:100
    image = imread(strcat('CarTestImages/test_car', sprintf('%03d',count),'.jpg'));
    harris = {harrisDetector(image,  3e11)};
    testFeatures = getPatches(harris, image, featureLength);
    [~,idx_test] = pdist2(C,[testFeatures.pixels]','euclidean','Smallest',1);
    
    % Thresholding
    votes = zeros(size(image));
    for j = 1:size(testFeatures,2)
        %locations= bsxfun(@minus, [testFeatures(j).location],  vocab(idx_test(j)).voteLocations);
        locations = zeros(size(vocab(idx_test(j)).voteLocations));
        for i = 1:size(vocab(idx_test(j)).voteLocations,1)
            locations(i,1) = testFeatures(j).location(1) - vocab(idx_test(j)).voteLocations(i,1);
            locations(i,2) = testFeatures(j).location(2) - vocab(idx_test(j)).voteLocations(i,2);
        end
        rows = round(locations(:,1));
        cols = round(locations(:,2));
        for k = 1:size(rows,1)
            if rows(k) > 0 && cols(k) > 0 && rows(k) < size(image,1) && cols(k) < size(image,2)
                votes(rows(k), cols(k)) = votes(rows(k), cols(k)) + 1;
            end
        end
    end
%     
%     filter = zeros(25,25);
%     filter(13,13) = 1;
%     filter = imgaussfilt(filter, 4);
%     votes = imfilter(votes, filter, 'replicate', 'full');
    filter = [1 2 1; 1 10 1; 1 2 1];
    votes = imfilter(votes, filter);
    max(max(votes))
    sum(sum(votes))
    threshold = max(max(votes));
    votesSorted = reshape(votes, size(votes,1)*size(votes,2),1);
    votesSorted = sort(unique(votesSorted), 'descend');
    results(count).locations = [];
    [row, col] = find(votes == max(max(votes)));
    results(count).locations = [[col(1)-colOffset,row(1)-rowOffset]];
    for i = 2:size(row,1)
        if max(pdist2([row, col], [row(i), col(i)],'euclidean','Smallest',2)) > 20
            results(count).locations = [results(count).locations; col(i)-colOffset, row(i)-rowOffset]; 
        end 
    end 
     
    results(count).truth = groundtruth(count).topLeftLocs;
    [~,closest] = pdist2(results(count).truth,[results(count).locations],'euclidean','Smallest',1);
    results(count).accuracy = [];
    results(count).correct = [];
    for i = 1:size(closest,2)
        [correct, accuracy] = testBox(100, 40, results(count).truth(closest(i),1),results(count).truth(closest(i),2), ...
            results(count).locations(i,1), results(count).locations(i,2));
        results(count).accuracy = [results(count).accuracy; accuracy];
        results(count).correct = [results(count).correct; correct];
    end
end
accuracy = [];
correct = []; 
for i=1:size(results,2)
    for j = 1:size(results(i).accuracy)
        accuracy = [accuracy, results(i).accuracy(j)];
        correct = [correct, results(i).correct(j)];
    end
end
mean(accuracy)
max(accuracy)
sum(correct) 

