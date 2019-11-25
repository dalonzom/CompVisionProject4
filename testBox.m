function [correct, IoU] = testBox(width, height, trueX, trueY, predictedX, predictedY)
I = intersection(width, height, trueX, trueY, predictedX, predictedY);
IoU = I/(2*width*height - I); 
correct = (IoU > 0.5); 
end