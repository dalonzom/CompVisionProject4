clear;
clc;
%% Training
%% Read in training images and use Harris corner detection
first = 1; 
last = 550;
featureLength = 7;
images = imread(strcat('CarTrainImages/train_car', sprintf('%03d',first),'.jpg'));
count = 1;
harris ={};
for i = first:last
    count = count + 1;
    images(:,:,count) = imread(strcat('CarTrainImages/train_car', sprintf('%03d',i),'.jpg'));
    harris{i} = {harrisDetector(images(:,:,count), 2e11)};
end

%% Extract 25x25 image patch for each feature
features = getPatches(harris, images, featureLength);

%% Use Kmeans to cluster the data
clusters = 300;
patches = zeros(size(features,2),size(features(1).pixels,1));
for i = 1:size(features,2)
    patches(i,:) = features(i).pixels';
end
[idx, C] = kmeans(patches,clusters, 'maxiter', 1000);


%% Assign local patches to words in vocabulary, record possible displacement
%% vectors between word and object center
rowOffset = 20;
colOffset = 50;
vocab = buildVocab(features, idx, clusters, C, rowOffset, colOffset);

%% Testing
load('GroundTruth/CarsGroundTruthBoundingBoxes.mat')
results = struct();
for count = 1:100
    count
    image = imread(strcat('CarTestImages/test_car', sprintf('%03d',count),'.jpg'));
    harris = {harrisDetector(image, 1e11)};
    testFeatures = getPatches(harris, image, featureLength);
    [vals,idx_test] = pdist2(C,[testFeatures.pixels]','euclidean','Smallest',1);
    
    % Thresholding
    votes = zeros(size(image));
    valTH = min(vals)*1.5;
    for j = 1:size(testFeatures,2)
        if vals(j) > valTH
            continue
        end
        %locations= bsxfun(@minus, [testFeatures(j).location],  vocab(idx_test(j)).voteLocations);
        locations = zeros(size(vocab(idx_test(j)).voteLocations));
        for i = 1:size(vocab(idx_test(j)).voteLocations,1)
            locations(i,1) = testFeatures(j).location(1) + vocab(idx_test(j)).voteLocations(i,1);
            locations(i,2) = testFeatures(j).location(2) + vocab(idx_test(j)).voteLocations(i,2);
        end
        rows = round(locations(:,1));
        cols = round(locations(:,2));
        for k = 1:size(rows,1)
            if rows(k) > 0 && cols(k) > 0 && rows(k) < size(image,1) && cols(k) < size(image,2)
                votes(rows(k), cols(k)) = votes(rows(k), cols(k)) + 1;
            end
        end
    end
    
    filter = zeros(7,7);
    filter(3:5,3:5) = 1;
    filter = imgaussfilt(filter, 2);
    %filter = ones(7);
    votes = imfilter(votes, filter, 'replicate', 'full');
    %votes = imfilter(votes, ones(5,5));
    max(max(votes));
    sum(sum(votes));
    threshold = max(max(votes));
    threshold = threshold * 0.97;
    votesSorted = reshape(votes, size(votes,1)*size(votes,2),1);
    [votesSorted, idx2] = sort(unique(votesSorted), 'descend');
    results(count).locations = [];
    x = 1;
    while size(results(count).locations,1) < 10
        if votesSorted(x) > threshold
            [row, col] = find(votes == votesSorted(x));
            results(count).locations = [results(count).locations; [col row]];
            x = x + 1;
        else
            break
        end
    end
    if size(results(count).locations,1) < 1
        results(count).accuracy = [0];
        results(count).correct = [0];
        continue
    end
    delI = [];
    for i = 2:size(results(count).locations,1)
        loc1 = results(count).locations(i,:);
        for j = 1:(i-1)
            loc2 = results(count).locations(j,:);
            if testBox(100, 40, loc1(2), loc1(1), loc2(2), loc2(1)) > 0.5
                delI = [delI i];
            end
        end
    end
    results(count).locations(delI,:) = [];
    results(count).truth = groundtruth(count).topLeftLocs;
    results(count).truth(:,1) = results(count).truth(:,1) + colOffset;
    results(count).truth(:,2) = results(count).truth(:,2) + rowOffset;
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
for i=1:size(results,2)
    for j = 1:size(results(i).accuracy)
        accuracy = [accuracy, results(i).accuracy(j)];
    end
end
mean(accuracy)
max(accuracy)
size(find(accuracy>=0.5))/size(accuracy)
