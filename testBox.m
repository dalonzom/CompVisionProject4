function [correct] = testBox(width, height, trueX, trueY, predictedX, predictedY)
I = intersection(width, height, trueX, trueY, predictedX, predictedY);
correct = (I/(2*width*height - I) > 0.5)
end