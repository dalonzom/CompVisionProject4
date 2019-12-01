function [vocab] = buildVocab(features, idx, clusters, C, rowOffset, colOffset)
vocab = struct();
for i=1:clusters
    vocab(i).mean = reshape(C(i,:),size([features(1).pixels]));
    vocab(i).displacments = [];
end

for i=1:size(idx,1)
    loc = features(i).location;
    loc(1) = rowOffset - loc(1);
    loc(2) = colOffset - loc(2);
%     if loc(1) <= rowOffset
%         loc(1) = loc(1) - rowOffset; 
%     else
%         loc(1) = loc(1)+rowOffset; 
%     end 
%     if loc(2) <= colOffset
%         loc(2) = loc(2) - colOffset; 
%     else
%         loc(2) = loc(2) + colOffset; 
%     end 
        
    vocab(idx(i)).displacments = [vocab(idx(i)).displacments; loc(1), loc(2)];
end
cValTH = 1500;
for i =1:size(vocab,2)
    i
    if size(vocab(i).displacments,1) > 5
         eval = evalclusters([vocab(i).displacments], 'kmeans', 'CalinskiHarabasz', 'klist', [1:30]);
         k = 1;
         if isnan(eval.OptimalK)
             eval.CriterionValues
         end
        [idx2, C, sumD] = kmeans([vocab(i).displacments], eval.OptimalK); 
        
%         cVals = eval.CriterionValues;
%          for j = 2:4
%              if cVals(j) < cValTH || ~(isnan(cVals(j) - cVals(j-1)) || cVals(j) - cVals(j-1) > cValTH)
%                  break
%              end
%              k = k + 1;
%          end
        
        vocab(i).voteLocations = C; 
    elseif size(vocab(i).displacments,1) ~= 1
        vocab(i).voteLocations = mean([vocab(i).displacments]); 
    else 
         vocab(i).voteLocations = vocab(i).displacments; 
    end 
        
end 
end 