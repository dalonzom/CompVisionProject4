function [vocab] = buildVocab(features, idx, clusters, C, rowOffset, colOffset)
vocab = struct();
for i=1:clusters
    vocab(i).mean = reshape(C(i,:),size([features(1).pixels]));
    vocab(i).displacments = [];
end

for i=1:size(idx,1)
    loc = features(i).location;
    vocab(idx(i)).displacments = [vocab(idx(i)).displacments; loc(1)-rowOffset, loc(2)-colOffset];
end

end 