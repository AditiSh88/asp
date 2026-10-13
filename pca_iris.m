% Iris dataset -> standardization -> pca -> visualization -> classification -> evaluation

clc;
clear;
close all;

rng(42);

load fisheriris

X = meas;                 % 150 x 4 feature matrix
Y = categorical(species); % Three species labels

featureNames = {'Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'};

classNames = categories(Y);

fprintf('Dataset dimensions: %d x %d\n', size(X,1), size(X,2));
disp('Class names:');
disp(classNames);

irisTable = array2table(X, 'VariableNames', {'SepalLength','SepalWidth','PetalLength','PetalWidth'});

irisTable.Species = Y;

disp('First 10 samples:');
disp(irisTable(1:10,:));

figure('Name','Original Features');

gscatter(X(:,3), X(:,4), Y);

xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
title('Iris Dataset: Original Feature Space');
grid on;

muAll = mean(X, 1);
sigmaAll = std(X, 0, 1);

Xscaled = (X - muAll) ./ sigmaAll;

% pca

[coeffAll, scoreAll, latentAll, ~, explainedAll] = pca(Xscaled);

X_pca = scoreAll(:,1:2);

fprintf('\nVariance explained by PC1: %.2f%%\n', explainedAll(1));
fprintf('Variance explained by PC2: %.2f%%\n', explainedAll(2));
fprintf('Total variance retained: %.2f%%\n', sum(explainedAll(1:2)));


figure('Name','PCA Visualization');

gscatter(X_pca(:,1), X_pca(:,2), Y);

xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('PCA Visualization of Iris Dataset');
legend('Location','best');
grid on;

cv = cvpartition(Y, 'HoldOut', 0.20);

trainIdx = training(cv);
testIdx = test(cv);

XTrain = X(trainIdx,:);
XTest = X(testIdx,:);

YTrain = Y(trainIdx);
YTest = Y(testIdx);

fprintf('\nTraining samples: %d\n', size(XTrain,1));
fprintf('Testing samples: %d\n', size(XTest,1));

muTrain = mean(XTrain, 1);
sigmaTrain = std(XTrain, 0, 1);

sigmaTrain(sigmaTrain == 0) = 1;

XTrainZ = (XTrain - muTrain) ./ sigmaTrain;
XTestZ = (XTest - muTrain) ./ sigmaTrain;

[coeffTrain, scoreTrain, ~, ~, explainedTrain, muPCA] = pca(XTrainZ);

XTrainPCA = scoreTrain(:,1:2);

XTestPCA = (XTestZ - muPCA) * coeffTrain(:,1:2);

fprintf('\nTraining PCA variance retained: %.2f%%\n', sum(explainedTrain(1:2)));

svmOriginal = templateSVM('KernelFunction','gaussian','Standardize',false);
modelOriginal = fitcecoc(XTrainZ, YTrain,'Learners', svmOriginal);
svmPCA = templateSVM('KernelFunction','gaussian','Standardize',false);
modelPCA = fitcecoc(XTrainPCA, YTrain, 'Learners', svmPCA);

predOriginal = predict(modelOriginal, XTestZ);
predPCA = predict(modelPCA, XTestPCA);

accuracyOriginal = mean(predOriginal == YTest) * 100;
accuracyPCA = mean(predPCA == YTest) * 100;

fprintf('\nOriginal features accuracy: %.2f%%\n', accuracyOriginal);
fprintf('PCA features accuracy: %.2f%%\n', accuracyPCA);

figure('Name','Original Feature Confusion Matrix');

confusionchart(YTest, predOriginal);
title('SVM: Original Four Features');

figure('Name','PCA Confusion Matrix');

confusionchart(YTest, predPCA);
title('SVM: Two PCA Components');

classList = categories(Y);

metricsOriginal = zeros(3,3);
metricsPCA = zeros(3,3);

for k = 1:numel(classList)

    actual = (YTest == classList{k});

    predictedOriginal = (predOriginal == classList{k});
    predictedPCA = (predPCA == classList{k});

    TP = sum(actual & predictedOriginal);
    FP = sum(~actual & predictedOriginal);
    FN = sum(actual & ~predictedOriginal);

    precision = TP / max(TP + FP, 1);
    recall = TP / max(TP + FN, 1);
    f1 = 2 * precision * recall / max(precision + recall, eps);

    metricsOriginal(k,:) = [precision, recall, f1];

    TP = sum(actual & predictedPCA);
    FP = sum(~actual & predictedPCA);
    FN = sum(actual & ~predictedPCA);

    precision = TP / max(TP + FP, 1);
    recall = TP / max(TP + FN, 1);
    f1 = 2 * precision * recall / max(precision + recall, eps);

    metricsPCA(k,:) = [precision, recall, f1];

end

macroOriginal = mean(metricsOriginal, 1);
macroPCA = mean(metricsPCA, 1);

fprintf('\nOriginal features - Macro Precision: %.3f\n', macroOriginal(1));
fprintf('Original features - Macro Recall: %.3f\n', macroOriginal(2));
fprintf('Original features - Macro F1: %.3f\n', macroOriginal(3));

fprintf('\nPCA features - Macro Precision: %.3f\n', macroPCA(1));
fprintf('PCA features - Macro Recall: %.3f\n', macroPCA(2));
fprintf('PCA features - Macro F1: %.3f\n', macroPCA(3));

Method = ["Original features (4D)"; "PCA features (2D)"];

Accuracy = [accuracyOriginal; accuracyPCA];

MacroPrecision = [macroOriginal(1); macroPCA(1)];
MacroRecall = [macroOriginal(2); macroPCA(2)];
MacroF1 = [macroOriginal(3); macroPCA(3)];

Results = table(Method, Accuracy, MacroPrecision, MacroRecall, MacroF1);

disp(' ');
disp('Final Classification Comparison:');
disp(Results);

figure('Name','Classification Accuracy Comparison');

bar(Accuracy);

set(gca, 'XTick', 1:2, 'XTickLabel', {'Original (4D)', 'PCA (2D)'});

ylabel('Test Accuracy (%)');
title('SVM Classification: Original Features vs PCA');
ylim([0 105]);
grid on;
