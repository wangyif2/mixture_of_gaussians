function [ avgErrTrain, avgErrTest ] = avgClErrRate(K,iter)
    load digits;

    %train on both digit 2 and 3 and get the best output model in 10 tries
    %(using K-means to initialize the mean vector)
    logProbX2 = -99999999;
    logProbX3 = -99999999;
    
    for iter1=1:10
        [p2temp,mu2temp,vary2temp,logProbX2temp] = mogEM(train2, K, iter, 0.01, 0);
        if(logProbX2temp>logProbX2)
            p2 = p2temp;
            mu2 = mu2temp;
            vary2 = vary2temp;
            logProbX2 = logProbX2temp;
        end
        
        [p3temp,mu3temp,vary3temp,logProbX3temp] = mogEM(train3, K, iter, 0.01, 0);
        if(logProbX3temp>logProbX3)
            p3 = p3temp;
            mu3 = mu3temp;
            vary3 = vary3temp;
            logProbX3 = logProbX3temp;
        end
    end

%     [p2,mu2,vary2,logProbX2] = mogEM(train2, K, iter, 0.01, 0);
%     [p3,mu3,vary3,logProbX3] = mogEM(train3, K, iter, 0.01, 0);

    %Compute the correctly classified training data for digit2
    [prob2, prob3] = mogClassification(p2,mu2,vary2,p3,mu3,vary3,train2);
    correctTrain2 = sum(prob2(1,:)>prob3(1,:));

    %Compute the correctly classified training data for digit3
    [prob2, prob3] = mogClassification(p2,mu2,vary2,p3,mu3,vary3,train3);
    correctTrain3 = sum(prob2(1,:)<prob3(1,:));

    %Compute the correctly classified test data for digit2
    [prob2, prob3] = mogClassification(p2,mu2,vary2,p3,mu3,vary3,test2);
    correctTest2 = sum(prob2(1,:)>prob3(1,:));

    %Compute the correctly classified test data for digit2
    [prob2, prob3] = mogClassification(p2,mu2,vary2,p3,mu3,vary3,test3);
    correctTest3 = sum(prob2(1,:)<prob3(1,:));
    
    %Compute the avg err rate
    avgErrTrain = (600-correctTrain2-correctTrain3)/600;
    avgErrTest = (600-correctTest2-correctTest3)/600;
end

