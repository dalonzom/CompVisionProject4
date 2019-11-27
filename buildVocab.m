function [vocab] = buildVocab(features, idx, clusters, C, rowOffset, colOffset)
vocab = struct();
for i=1:clusters
    vocab(i).mean = reshape(C(i,:),size([features(1).pixels]));
    vocab(i).displacments = [];
end

for i=1:size(idx,1)
    loc = features(i).location;
    if loc(1) <= rowOffset
        loc(1) = loc(1) - rowOffset; 
    else
        loc(1) = loc(1)+rowOffset; 
    end 
    if loc(2) <= colOffset
        loc(2) = loc(2) - colOffset; 
    else
        loc(2) = loc(2) + colOffset; 
    end 
        
    vocab(idx(i)).displacments = [vocab(idx(i)).displacments; loc(1), loc(2)];
end
for i =1:size(vocab,2)
    if size(vocab(i).displacments,1) > 10
         eval = evalclusters([vocab(i).displacments], 'kmeans', 'gap', 'klist',[1:4]);
        [idx, C] = kmeans([vocab(i).displacments], eval.OptimalK); 
        vocab(i).voteLocations = C; 
    elseif size(vocab(i).displacments,1) ~= 1
        vocab(i).voteLocations = mean([vocab(i).displacments]); 
    else 
         vocab(i).voteLocations = vocab(i).displacments; 
    end 
        
end 
end 