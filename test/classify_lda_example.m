%% Example code to use LDA classifier
clc;clearvars;close all;

npat = 1000;

% %% create synthetic data
% disp('Creating new data');
% r1 = 0.5 + 0.95*randn(npat,5);
% r2 = 1.5 + 0.95*randn(npat*2,5);
% data = [r1;r2];
% labels = [ones(1,size(r1,1)) 2*ones(1,size(r2,1))];
% 
% %% Save data
% disp('Save the data')
% writematrix(data, '/home/paolo/rosneuro_ws/src/rosneuro_decoder_lda/test/features.csv');
% writematrix(labels, '/home/paolo/rosneuro_ws/src/rosneuro_decoder_lda/test/labels.csv');

%% load data
disp('load data')
load('/home/paolo/rosneuro_ws/src/rosneuro_decoder_lda/test/features.csv')
data = features;
load('/home/paolo/rosneuro_ws/src/rosneuro_decoder_lda/test/labels.csv')

%% train LDA
model = classify_lda_train(data,labels,'lda');

%% test LDA
[pp, res] = classify_lda_eval(model,data);

accuracy  = 100*sum(res'==labels)/length(labels);

%% graphics
figure(1)
plot(data(labels==1,1),data(labels==1,2),'.') ; hold on;
plot(data(labels==2,1),data(labels==2,2),'gx');
axis([0 6 0 6]);
mis_ind = find(res'~=labels);
plot(data(mis_ind,1),data(mis_ind,2),'ro','markersize',10);
legend('class-1', 'class-2', 'misclassified');
hold off
disp(['accuracy: ' num2str(accuracy)]);

%% Load rosneuro probabilities
disp('load the computed raw probability of rosneuro lda decoder')
load('/home/paolo/rosneuro_ws/src/rosneuro_decoder_lda/test/output.csv');

%% See differences
diff = max(abs(pp - output), [], 'all');
disp(['max difference: ' num2str(diff)]);