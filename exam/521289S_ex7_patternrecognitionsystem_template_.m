function patternRecognitionSystem(X_features, X_classes, k)
% Pattern recognition system
% OUTPUTS:
% INPUTS:
% X_features       matrix containing the data
% X_classes        correct labels for the data

% Do the following first:
%
% % Clear and close all
% close all;
% clear;
% clc;
% 
% % Load data
 load e7_data.mat
% 
 X_features = X(:, 2:end);
 X_classes = X(:,1);
%
 k = 1; % 1-NN

N = size(X_features,1);
num_features = size(X_features,2);

% Whitening the data
X_stand = standardize(X_features); %Not implemented. This is your first task.

% Select training and validation sets
% Forward search implements leave-one-out cross validation therefore a separate test set is
% included in training
% 2/3 for training and 1/3 for validation. Use randperm for suffling the data, then divide it into sets. 

% Train feature vector
fvector = zeros(num_features,1);
best_result = 0;
for in = 1:num_features
    [best_result_add, best_feature_add] = forwardsearch(training_data, training_class, fvector, k);
     % Update the feature vector  
    fvector(best_feature_add) = 1;
  
    % Save best result
    if(best_result < best_result_add)
        best_result = best_result_add;
        best_fvector = fvector;
    end
    
end

best_result
best_fvector

% Test results. Train the system and evaluate the accuracy.


end

function [feat_out] =standardize(feat_in)


N = size(feat_in,1); 

% Method 1: Standardization


% Method 2: Whiten with eigenvalue decomposition
%see help eig, cov

% Whitening can be also done with SVD, as an optional task 



end

function [predictedLabels] = knnclass(dat1, dat2, fvec, classes, k)

    p1 = pdist2( dat1(:,logical(fvec)), dat2(:,logical(fvec)) );
    % Here we aim in finding k-smallest elements
    [D, I] = sort(p1', 1);

    I = I(1:k+1, :);
    labels = classes( : )';
    if k == 1 % this is for k-NN, k = 1
        predictedLabels = labels( I(2, : ) )';
    else % this is for k-NN, other odd k larger than 1
        predictedLabels = mode( labels( I( 1+(1:k), : ) ), 1)'; % see help mode
    end

end

function [best, feature] = forwardsearch(data, data_c, fvector, k)
    % SFS, from previous lesson.
    num_samples = length(data);
    best = 0;
    feature = 0;
    
    for in = 1:length(fvector)
        if (fvector(in) == 0)
            fvector(in) = 1;
            % Classify using k-NN
	        predictedLabels = knnclass(data, data, fvector, data_c, k);
            correct = sum(predictedLabels == data_c); % the number of correct predictions
            result = correct/num_samples; % accuracy
            if(result > best)
                best = result; 
                feature = in; 
            end
            fvector(in) = 0;
        end
    end

end

