%%=================== MACHINE LEARNING ASSIGNMENT 2 =======================
%
%   Team Members
%   Pavan Siva Kumar Amarapalli
%   Venkata Praneeth Bavirisetty
%   Shiva Vamsi Gudivada
%   Anuj Jain
%   
%   normalEq.m
%   computeCost.m
%   plotContour.m
%%=========================== Initialization ==============================
clear ;
close all;
clc

%% read in the data set
%The given dataset is having 97 training examples
dataFull = importdata('dow_jones_index.data');
stockData = dataFull.textdata;
X = stockData(2:751,3:13);     %Extracting input observations from the data
Y = stockData(:,4);       %Extracting output observations from the data
%%====================== Standardization of data ==========================
muX = mean(X);
stdX = std(X);

repstd = repmat (stdX, 97, 1);
repmu = repmat (muX, 97, 1);

standardizedX = (X - repmu)./repstd;  %Standardizing the data

mean(standardizedX);
std(standardizedX);

%We are not using standardized data as we are prefering normal equation
%method to find weights.


%Extended input (adding a column with all 1's to existing observation set)
eX = [ones(97, 1) X];

%%======================= Part1: L2-Regularization ========================
%dividing the data set into two subsets of approximately equal sizes
%TrainSet will have 49 training examples
%TestSet will have the remaining 48 training examples

TrainSetX = eX(1:49, :);    %Trainset input
TestSetX = eX(50:97, :);    %Testset input

TrainSetY = Y(1:49, :);     %Trainset output
TestSetY = Y(50:97, :);     %Testset output

%Using Normal Equation to obtain the value of lambda
[w, Jw, lambdaFinal] = normalEq(TrainSetX, TestSetX, TrainSetY, TestSetY);

%obtaining the smallest component in the weight vector
%to eliminate one attribute
[M, I] = min(w);
fprintf('The Lambda value obtained for minimum cost by using normal equation is %f\n\n', lambdaFinal);

%eliminating the attribue column corresponding to the smallest weight
%in weight vector
reducedX = [eX(:,1:I-1) eX(:,I+1:4)];

%%====================== Part 2: Cross-validation =========================
%performing a k-fold Cross-validation where k = 3
%dividing the dataset into 3 subsets
X1 = reducedX(1:33, :);     %subset 1
Y1 = Y(1:33, :);
X2 = reducedX(34:65, :);    %subset 2
Y2 = Y(34:65, :);
X3 = reducedX(66:97, :);    %subset 3
Y3 = Y(66:97, :);

%Performing cross validation for degree 1
[w12, Jwtrain1, Jw12] = computeCost(X1, X2, Y1, Y2);    %TrainSet = X1, %TestSet = X2
[w13, Jwtrain2, Jw13] = computeCost(X1, X3, Y1, Y3);    %TrainSet = X1, %TestSet = X3
[w21, Jwtrain3, Jw21] = computeCost(X2, X1, Y2, Y1);    %TrainSet = X2, %TestSet = X1
[w23, Jwtrain4, Jw23] = computeCost(X2, X3, Y2, Y3);    %TrainSet = X2, %TestSet = X3
[w31, Jwtrain5, Jw31] = computeCost(X3, X1, Y3, Y1);    %TrainSet = X3, %TestSet = X1
[W32, Jwtrain6, Jw32] = computeCost(X3, X2, Y3, Y2);    %TrainSet = X3, %TestSet = X2

%calculating the Average Cost for degree 1
AvgJw1 = (Jw12+Jw13+Jw21+Jw23+Jw31+Jw32)/6;
fprintf('Average Error for degree 1 is %f\n', AvgJw1);

%Performing cross validation for degree 2
d2reducedX= [reducedX reducedX(:, 2).*reducedX(:, 3) reducedX(:, 2).^2 reducedX(:, 3).^2];

X1 = d2reducedX(1:33, :);   %subset 1
Y1 = Y(1:33, :);
X2 = d2reducedX(34:65, :);  %subset 2
Y2 = Y(34:65, :);
X3 = d2reducedX(66:97, :);  %subset 3
Y3 = Y(66:97, :);

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

%Performing 100 iterations
JwTrainItr = [];
JwTestItr = [];

for iterationCount = 1:1:100                %loop for 100 times
    %creating random subsets of equal size
    c=randperm(97);
    randTrainX = d2reducedX (c(1:49),:);    %Random Train Set
    randTrainY = Y (c(1:49),:);
    randTestX = d2reducedX(c(50:97),:);     %Random Test Set
    randTestY = Y (c(50:97),:);
    %Computing Cost function for each random pair
    [wTemp,JwTemp1,JwTemp2] = computeCost(randTrainX, randTestX, randTrainY, randTestY);
    JwTrainItr = [JwTrainItr; JwTemp1];
    JwTestItr = [JwTestItr; JwTemp2];
end

fprintf('After 100 iterations\n\n')

%extracting the minimum Modelling Error of the 100 iterations
minJtrain = min(JwTrainItr);
fprintf('Minimum Modelling Error is %f\n', minJtrain)

%extracting the maximum Modelling Error of the 100 iterations
maxJtrain = max(JwTrainItr);
fprintf('Maximum Modelling Error is %f\n', maxJtrain)

%extracting the average Modelling Error of the 100 iterations
AvgJtrain=  mean(JwTrainItr);
fprintf('Average Modelling Error is %f\n\n', AvgJtrain)

%extracting the minimum Generalization Error of the 100 iterations
minJtest = min(JwTestItr);
fprintf('Minimum Generalization Error is %f\n', minJtest)

%extracting the minimum Generalization Error of the 100 iterations
maxJtest = max(JwTestItr);
fprintf('Maximum Generalization Error is %f\n', maxJtest)

%extracting the minimum Generalization Error of the 100 iterations
AvgJtest=  mean(JwTestItr);
fprintf('Average Generalization Error is %f\n', AvgJtest)

%%=================== Visualizing the output data =========================
set(gcf, 'Position', [20, 20, 1900, 960]);
p1 = plot(JwTrainItr);
set (p1,'LineStyle','-','Marker','o','MarkerFaceColor','red','MarkerSize',4,'Color','red','LineWidth',2);

hold all;
p2 = plot(JwTestItr);
set (p2,'LineStyle','-','Marker','o','MarkerFaceColor','blue','MarkerSize',4,'Color','blue','LineWidth',2);
legend('Modal Error','Generalization Error');
xlabel('Iteration'); ylabel('Error');

figure;
syms w0 w1 w2 w3 w4 w5;
w_s = [w0;w1;w2;w3;w4;w5];
w0 = wTemp(1);
w1 = wTemp(2);
w2 = wTemp(3);
w3 = wTemp(4);
w4 = wTemp(5);
w5 = wTemp(6);
xlength = length(randTrainX(:,1));
%Jw_new = sum((hX - y').^2);
%grad=gradient(Jw_new);
%[w0, w1, w2, w3, w4, w5]=solve(grad(1), grad(2), grad(3), grad(4), grad(5), grad(6));
for l=1:xlength, hX(l) = randTrainX(l,:)*w_s; end
plot(randTrainX(:,6), eval(hX), 'ro');
hold on;
plot(randTrainX(:,6), eval(hX), '-');

figure;             %adding new figure Window
set(gcf, 'Position', [20, 20, 1900, 960]);
hold all;

subplot(2,3,1);     %attribute x1
%sorting X & Y to plot hypothesis
[sortedX, sortIndex] = sort(randTrainX);
sortedY = randTrainY(sortIndex(:,1));
plot (sortedX(:,1), randTrainX*wTemp,'-');hold all;
plot (sortedX(:,1), sortedY,'r*');
title('Linear Regression');xlabel('feature x1'); ylabel('output y');
legend('hypothesis')

subplot(2,3,2);     %attribute x2
[sortedX, sortIndex] = sort(randTrainX);
sortedY = randTrainY(sortIndex(:,2));
plot (sortedX(:,2), randTrainX*wTemp,'-');hold all;
plot (sortedX(:,2), sortedY,'r*');
title('Linear Regression');xlabel('feature x2'); ylabel('output y');
legend('hypothesis')

subplot(2,3,3);     %attribute x3
[sortedX, sortIndex] = sort(randTrainX);
sortedY = randTrainY(sortIndex(:,3));
plot (sortedX(:,3), randTrainX*wTemp,'-');hold all;
plot (sortedX(:,3), sortedY,'r*');
title('Linear Regression');xlabel('feature x3'); ylabel('output y');
legend('hypothesis')

subplot(2,3,4);     %attribute x4
[sortedX, sortIndex] = sort(randTrainX);
sortedY = randTrainY(sortIndex(:,4));
plot (sortedX(:,4), randTrainX*wTemp,'-');hold all;
plot (sortedX(:,4), sortedY,'r*');
title('Linear Regression');xlabel('feature x4'); ylabel('output y');
legend('hypothesis')

subplot(2,3,5);     %attribute x5
[sortedX, sortIndex] = sort(randTrainX);
sortedY = randTrainY(sortIndex(:,5));
plot (sortedX(:,5), randTrainX*wTemp,'-');hold all;
plot (sortedX(:,5), sortedY,'r*');
title('Linear Regression');xlabel('feature x5'); ylabel('output y');
legend('hypothesis')

subplot(2,3,6);     %attribute x6
[sortedX, sortIndex] = sort(randTrainX);
sortedY = randTrainY(sortIndex(:,6));
plot (randTrainX(:,6), randTrainX*wTemp,'-');hold all;
plot (sortedX(:,6), sortedY,'r*');
title('Linear Regression');xlabel('feature x6'); ylabel('output y');
legend('hypothesis')

figure;             %adding new figure Window
set(gcf, 'Position', [20, 20, 1900, 960]);
%obtaining the cost function for different values of weights W1 & W2
%using the Train set of 100th iteration for plotting the graphs
[W1_vals, W2_vals, J_vals] = plotContour(randTrainX, randTrainY, wTemp);

subplot(1,2,1);     %surface plot
surf(W1_vals, W2_vals, J_vals);
title('Surface Plot');xlabel('W1'); ylabel('W2');zlabel('Cost Function Jw');

subplot(1,2,2);     %contour plot
p3 = contour(W1_vals, W2_vals, J_vals, logspace(-2, 3, 20));
title('Contour Plot');xlabel('W1'); ylabel('W2');