function [ mse ] = rootMeanSquareError( ActualY, predictedY )
%MEANSQUAREERROR Summary of this function goes here
%   Detailed explanation goes here
    Error = (ActualY - predictedY);
    mse = 0;
    for i = 1:length(ActualY)
        mse = mse + Error(i)*Error(i);
    end
    mse = sqrt(mse/length(ActualY));
end

