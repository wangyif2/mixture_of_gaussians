% function [ probd1, probd2 ] = mogClassification(p2,mu2,vary2,p3,mu3,vary3,x)
%
% Compute P(d=1|x) P(d=2|x) given mixture model parameters obtained from
% training previously
%
% Input:
%
%   Model parameters (for both digit 2, and 3):
%
%    p(k) = prior probability of the kth cluster
%    mu(n,k) = the nth component of the mean for the kth cluster
%    vary(j,k) = variance of the jth dimension in the kth cluster
%
%   Data:
%
%    x(n,t) = the nth input for the tth training case
%  
% Output:
%
%    logProb(t) = the log-probability of the tth case using the current model
%

function [ probd1, probd2 ] = mogClassification(p2,mu2,vary2,p3,mu3,vary3,x)

%feed the test image into model for digit2 and digit 3 seperately and
%obtain the log likelihood
[unnormalizedLog2] = mogLogProb(p2,mu2,vary2,x);
[unnormalizedLog3] = mogLogProb(p3,mu3,vary3,x);

%nomalize each log likelihood
[normalizedLog2,L2] = normalizeLogspace(unnormalizedLog2);
[normalizedLog3,L3] = normalizeLogspace(unnormalizedLog3);

%take the exponential
probd1(1,:) = exp(normalizedLog2(1,:));
probd2(1,:) = exp(normalizedLog3(1,:));

end

