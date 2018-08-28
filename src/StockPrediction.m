%%====================== MACHINE LEARNING PROJECT =========================
%
%   Team Members
%   Pavan Siva Kumar Amarapalli
%   Venkata Praneeth Bavirisetty
%   Shiva Vamsi Gudivada
%   Anuj Jain
%   
%   Functions used:
%   assignNumbersToSymbols.m
%   normalEq.m
%   computeCost.m
%   gradientDescentB.m
%   computeCostB.m
%   rootMeanSquareError.m
%   regSVM.m    
%   
%%=========================== Initialization ==============================

clear ;
close all;
clc

dataFull = importdata('dow_jones_index.data');	%importing data
%The Dow Jones Industrial Average is a collection of 30 publicly traded Companies
stockData = dataFull.textdata;
% substituting the words in the actual data with numbers
stockData = assignNumbersToSymbols(stockData);

Xfeatures = stockData(1,[1:11]);
% The built-in function strip is used to remove special characters present in the imported data
% The built-in function cellstr is used to typecast from cell to string, 
% which is eventually converted to double datatype using the str2double function
X = str2double(cellstr(strip(stockData(2:751,[1:2 4:11]),'left','$')));
Y = str2double(cellstr(strip(stockData(2:751,12),'left','$')));

sizeX = size(X);

%======================== Handling Missing data ===========================

%missing values are filled by taking the average for the particular company

Avg_per(30) = 0;		%Average of 'percent_change_volume_over_last_wk' for each company
Avg_vol(30) = 0;		%Average of 'previous_weeks_volume' for each company

for j = 1:30
    c_per = 0;
    c_vol = 0;
    for i = 1:sizeX(1)
        if isequal(X(i,2),j)
            if (~isnan(X(i,9)))
              Avg_per(j) = Avg_per(j) + X(i,9);
              c_per = c_per+1;
            end
            if (~isnan(X(i,10)))
              Avg_vol(j) = Avg_vol(j) + X(i,10);
              c_vol = c_vol+1;
            end
        end
    end
    Avg_per(j) = Avg_per(j)/c_per;
    Avg_vol(j) = Avg_vol(j)/c_vol;
end    

for i = 1:sizeX(1)
    for j = 1:sizeX(2)
        if ( isnan(X(i,j)))		%identify the empty cells
            if isequal(j,9)
                X(i,j) = Avg_per(rem(i,30));	%filling the empty cells with Average
            elseif isequal(j,10)
                X(i,j) = Avg_vol(rem(i,30));
            end    
        end    
    end
end

%========================Standardizing the data===========================%

muX = mean(X);
stdX = std(X);

repstd = repmat (stdX, sizeX(1), 1);
repmu = repmat (muX, sizeX(1), 1);

standardizedX = (X - repmu)./repstd;  %

eX = [ones(sizeX(1), 1) X];

% creating random subsets of equal size
c = randperm(750);
TrainSetX = eX (c(1:375),:);     %Random Train Set
TrainSetY = Y (c(1:375),:);
TestSetX = eX(c(376:750),:);     %Random Test Set
TestSetY = Y (c(376:750),:);

%%=========================== L2-Regularization ===========================
[W_normal, Jw, lambdaFinal] = normalEq(TrainSetX, TestSetX, TrainSetY, TestSetY);
[M, I] = min(abs(W_normal));

% eliminating the attribue column corresponding to the smallest weight
% in weight vector
reducedX = [eX(:,1:I-1) eX(:,I+1:length(eX(1,:)))];
fprintf('The following feature has been eliminated: %s \n', Xfeatures{I});
%celldisp(Xfeatures(I));

W_regularized = [W_normal(1:I-1, 1); W_normal(I+1:length(eX(1,:)), 1)];
Y_regularized = reducedX*W_regularized;

%%========================== Cross-validation =============================
% performing a k-fold Cross-validation where k = 3
% dividing the dataset into 3 subsets

X1 = reducedX(1:250, :);      %subset 1
Y1 = Y(1:250, :);
X2 = reducedX(251:500, :);    %subset 2
Y2 = Y(251:500, :);
X3 = reducedX(501:750, :);    %subset 3
Y3 = Y(501:750, :);

% Performing cross validation for degree 1
[~, ~, Jw12] = computeCost(X1, X2, Y1, Y2);    %TrainSet = X1, %TestSet = X2
[~, ~, Jw13] = computeCost(X1, X3, Y1, Y3);    %TrainSet = X1, %TestSet = X3
[~, ~, Jw21] = computeCost(X2, X1, Y2, Y1);    %TrainSet = X2, %TestSet = X1
[~, ~, Jw23] = computeCost(X2, X3, Y2, Y3);    %TrainSet = X2, %TestSet = X3
[~, ~, Jw31] = computeCost(X3, X1, Y3, Y1);    %TrainSet = X3, %TestSet = X1
[~, ~, Jw32] = computeCost(X3, X2, Y3, Y2);    %TrainSet = X3, %TestSet = X2

% calculating the Average Cost for degree 1
AvgJw1 = (Jw12+Jw13+Jw21+Jw23+Jw31+Jw32)/6;
fprintf('Average Error for degree 1 is %f\n', AvgJw1);

% Performing cross validation for degree 2
d2reducedX = [reducedX reducedX(:, 3).^2];

X1 = d2reducedX(1:250, :);    %subset 1
Y1 = Y(1:250, :);
X2 = d2reducedX(251:500, :);  %subset 2
Y2 = Y(251:500, :);
X3 = d2reducedX(501:750, :);  %subset 3
Y3 = Y(501:750, :);

[w12, Jwtrain1, Jw12] = computeCost(X1, X2, Y1, Y2);    %TrainSet = X1, %TestSet = X2
[w13, Jwtrain2, Jw13] = computeCost(X1, X3, Y1, Y3);    %TrainSet = X1, %TestSet = X3
[w21, Jwtrain3, Jw21] = computeCost(X2, X1, Y2, Y1);    %TrainSet = X2, %TestSet = X1
[w23, Jwtrain4, Jw23] = computeCost(X2, X3, Y2, Y3);    %TrainSet = X2, %TestSet = X3
[w31, Jwtrain5, Jw31] = computeCost(X3, X1, Y3, Y1);    %TrainSet = X3, %TestSet = X1
[W32, Jwtrain6, Jw32] = computeCost(X3, X2, Y3, Y2);    %TrainSet = X3, %TestSet = X2

%calculating the Average Cost for degree 2
AvgJw2 = (Jw12+Jw13+Jw21+Jw23+Jw31+Jw32)/6;
fprintf('Average Error for degree 2 is %f\n', AvgJw2);
fprintf('Average Error is minimum for degree 2 in cross validation\n\n');

[W_d2, Jw, lambdaFinal] = normalEq(d2reducedX, d2reducedX, Y, Y);

Y_d2 = d2reducedX*W_d2;

c = randperm(750);
d2TrainSetX = d2reducedX (c(1:375),:);     %Random Train Set
d2TrainSetY = Y (c(1:375),:);
d2TestSetX = d2reducedX(c(376:750),:);     %Random Test Set
d2TestSetY = Y (c(376:750),:);

%%=========================== Gradient Descent ============================

eX_g = [ones(sizeX(1), 1) standardizedX];
dgTrainSetX = eX_g (c(1:375),:);
dgTrainSetY = Y (c(1:375),:);
dgTestSetX = eX_g (c(376:750),:);
dgTestSetY = Y (c(376:750),:);
W = zeros(length(eX_g(1,:)), 1); 
alpha = 0.1;
num_iters = 50000;
[W_gradient, J_gradient] = gradientDescentB(dgTrainSetX, dgTrainSetY, W, alpha, num_iters);

Y_gradient = dgTestSetX*W_gradient;

fprintf('Root Mean Square Error for Gradient Descent is %f\n', rootMeanSquareError( dgTestSetY, Y_gradient));

%%=========================== Normal Equation =============================

[W_normal, Jw, lambdaFinal] = normalEq(d2TrainSetX, d2TestSetX, d2TrainSetY, d2TestSetY);

Y_normal = d2TestSetX*W_normal;

fprintf('Root Mean Square Error for Normal Equation is %f\n', rootMeanSquareError( d2TestSetY, Y_normal));

%%============================ Regression SVM =============================
SVM_val = regSVM (d2reducedX, Y, 'Standardize', true);

Y_rsvm = predict(SVM_val,d2TestSetX);
fprintf('Root Mean Square Error for Support Vector Regression is %f\n', rootMeanSquareError( d2TestSetY, Y_rsvm));

%%=========================== Regression Tree =============================
tree_val = fitrtree (d2reducedX, Y);
Y_tree = predict(tree_val,d2TestSetX);
fprintf('Root Mean Square Error for Regression Tree is %f\n', rootMeanSquareError( d2TestSetY, Y_tree));

%%======================== Visualizing the data ===========================

set(gcf, 'Position', [10, 0, 1500, 780]);

subplot(2,2,1)
title('IBM');xlabel('Sample Data'); ylabel('Stock Price');
hold on;
le = find(d2TestSetX(:,3) == 13);
plot (1:length(le),d2TestSetY(le),'k','DisplayName','Actual Output')
plot (1:length(le),Y_gradient(le),'g','LineStyle','- -','DisplayName','Gradient Descent')
plot (1:length(le),Y_normal(le),'b','LineStyle','- -','DisplayName','Normal Equation')
plot (1:length(le),Y_rsvm(le),'r','LineStyle','- -','DisplayName','Regression SVM')
plot (1:length(le),Y_tree(le),'c','LineStyle','- -','DisplayName','Regression Tree')
legend('show');

subplot(2,2,2)
title('Bank of America Corporation(BAC)');xlabel('Sample Data'); ylabel('Stock Price');
hold on;
le = find(d2TestSetX(:,3) == 4);
plot (1:length(le),d2TestSetY(le),'k','DisplayName','Actual Output')
plot (1:length(le),Y_gradient(le),'g','LineStyle','- -','DisplayName','Gradient Descent')
plot (1:length(le),Y_normal(le),'b','LineStyle','- -','DisplayName','Normal Equation')
plot (1:length(le),Y_rsvm(le),'r','LineStyle','- -','DisplayName','Regression SVM')
plot (1:length(le),Y_tree(le),'c','LineStyle','- -','DisplayName','Regression Tree')
legend('show');

subplot(2,2,3)
title('Microsoft (MSFT)');xlabel('Sample Data'); ylabel('Stock Price');
hold on;
le = find(d2TestSetX(:,3) == 22);
plot (1:length(le),d2TestSetY(le),'k','DisplayName','Actual Output')
plot (1:length(le),Y_gradient(le),'g','LineStyle','- -','DisplayName','Gradient Descent')
plot (1:length(le),Y_normal(le),'b','LineStyle','- -','DisplayName','Normal Equation')
plot (1:length(le),Y_rsvm(le),'r','LineStyle','- -','DisplayName','Regression SVM')
plot (1:length(le),Y_tree(le),'c','LineStyle','- -','DisplayName','Regression Tree')
legend('show');

subplot(2,2,4)
title('McDonalds (MCD)');xlabel('Sample Data'); ylabel('Stock Price');
hold on;
le = find(d2TestSetX(:,3) == 19);
plot (1:length(le),d2TestSetY(le),'k','DisplayName','Actual Output')
plot (1:length(le),Y_gradient(le),'g','LineStyle','- -','DisplayName','Gradient Descent')
plot (1:length(le),Y_normal(le),'b','LineStyle','- -','DisplayName','Normal Equation')
plot (1:length(le),Y_rsvm(le),'r','LineStyle','- -','DisplayName','Regression SVM')
plot (1:length(le),Y_tree(le),'c','LineStyle','- -','DisplayName','Regression Tree')
legend('show');