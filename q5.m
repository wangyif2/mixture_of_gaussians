clear all
load digits

K = [2,5,15,20,25];
numIter = size(K,2);
avgErrTrain = zeros(numIter,1);
avgErrTest = zeros(numIter,1);

for iter=1:numIter
    [avgErrTrain(iter,1), avgErrTest(iter,1)] = avgClErrRate(K(1,iter), 30);    
end



figure;
hold on;

plot(K,avgErrTrain,'r-',K,avgErrTest,'b--');

title('avg classification error rate VS num cluster');
xlabel('number of cluster');
ylabel('avg classification error rate');
legend('Avg Err Train','Avg Err Test');

