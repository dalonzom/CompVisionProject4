function [intersection] = intersection(width, height, x1, y1, x2, y2)
intersection = max(0,height-abs(y2-y1))*max(0,width-abs(x2-x1));
end