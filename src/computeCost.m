function [w JwTrain JwTest] = computeCost(TrainSetX, TestSetX, TrainSetY, TestSetY)
    %computeCost calculates the valuse of Jw for a given data set
    m1 = length(TrainSetX(:, 1));    %extract the number of training examples
    m2 = length(TestSetX(:, 1));     %extract the number of test examples
    
    %calculating the Weight Vector
    w = (pinv(TrainSetX'*TrainSetX))*TrainSetX'*TrainSetY;
    
    %hX holds the values for hypothesis function
    %evaluating hypothesis for Train data set
    for l=1:m1, hXTrain(l) = TrainSetX(l,:)*w; end
    %evaluating hypothesis for Test data set
    for l=1:m2, hXTest(l) = TestSetX(l,:)*w; end
    
    %calculating cost for Train data set
    JwTrain = (1/(2*m1))*(sum((hXTrain-TrainSetY').^2));
    %calculating cost for Train data set
    JwTest = (1/(2*m2))*(sum((hXTest-TestSetY').^2));
end