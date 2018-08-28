function [w Jw,lambdaFinal]= normalEq(TrainSetX, TestSetX, TrainSetY, TestSetY)
    %normalEq calculates the Weight Vector and cost for a given data set
    m1 = length(TrainSetX(1, :));    %extract the number of training examples
    m2 = length(TestSetX(:, 1));     %extract the number of test examples
    
    TempL = [];                     %temporary variable to store values of Jw and lambda
    L = [zeros(1, m1) ; zeros(m1-1, 1) eye(m1-1)]; 
    %calculating the cost for different values of lambda
    for lambda = 0.01:0.01:10
        w = (pinv(TrainSetX'*TrainSetX  + lambda*L))*TrainSetX'*TrainSetY;
        for l = 1:m2, hX(l) = TestSetX(l, :)*w; end
        %computing the cost
        Jw = (1/(2*m2))*(sum((hX-TestSetY').^2));
        %storing the values in temporary variable
        TempL = [TempL; lambda, Jw];
    end
    %Obtaining the value of lambda where Jw is minimum
    [~, I] = min(TempL(:, 2));
    lambdaFinal = TempL(I, 1);
    Jw = TempL(I, 2);               %cost for minimum value of lambda
    %weight vector for minimum value of lambda
    w = (pinv(TrainSetX'*TrainSetX  + lambdaFinal*L))*TrainSetX'*TrainSetY;
end