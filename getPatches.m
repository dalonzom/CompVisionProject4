function [features] = getPatches(harris, images, length)
features = struct();
count = 1; 
for i = 1:size(harris,2)
    if strcmp(class(harris{i}), 'cell')
        mat = cell2mat(harris{i});
    else 
        mat = harris{i}; 
    end 
    image = images(:,:,:,i); 
    [rows, columns, ~] = find(mat ~= 0); 
    for j = 1:size(rows,1)
        row = rows(j); 
        col = columns(j);
        image = padarray(image,[length length],0,'both');
        %if row > length && col > length && row < size(mat,1)-length && col < size(mat,2)-length 
            vec = image(row:row+2*length, col:col+2*length);
            features(count).pixels = reshape(vec, size(vec,1)*size(vec,2),1); 
            features(count).location = [row-length, col-length]; 
            count = count + 1; 
       % end
    end 
    
end
end 