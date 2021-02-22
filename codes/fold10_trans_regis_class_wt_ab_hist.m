correct=0;
correct_list=zeros(1,124);
rng(1);
load('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/normal_abnormal_ent_log_Errors.mat');
% load('\\medctr.ad.wfubmc.edu\dfs\cialab$\scratch\cialab\scamalan\scamalan\Autoscope\Pair\normal_abnormal_ent_log_Errors.mat');

addpath('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope');
% addpath('\\medctr.ad.wfubmc.edu\dfs\cialab$\scratch\cialab\scamalan\Autoscope');
location = '/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/ENT';
% location = '\\medctr.ad.wfubmc.edu\dfs\cialab$\scratch\cialab\scamalan\Autoscope\Pair\Normal_Abnormal\ENT';
imds = imageDatastore(location,'LabelSource','foldernames','IncludeSubfolders',true);
folder = 'w_reg/' ;  %'wo_reg/' %'w_reg/'  %'reg_hist_std/' %'reg_hist_std-mean/'  %'reg_hist/'
     
p=124;
n = [1;1]*(1:p);
idx = n(:)';
k=10;
ten = ones(10,1);       five = ones(5,1);
twelve = ones(12,1);    six = ones(6,1);
forteen = ones(14,1);   seven = ones(7,1);
sixteen = ones(16,1);   eight = ones(8,1);
fold = [reshape(ten*(1:7),[1 70]),reshape(twelve*(8:10),[1 36]),...
    reshape(forteen*(1:9),[1 126]),reshape(sixteen*(10:10),[1 16])];
half_fold = [reshape(five*(1:7),[1 35]),reshape(six*(8:10),[1 18]),...
    reshape(seven*(1:9),[1 63]),reshape(eight*(10:10),[1 8])];

for i=1:k
    test_idx = (fold == i);
    train_idx = ~test_idx;
    imdsTest = imageDatastore(imds.Files(fold == i), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    imdsTrain = imageDatastore(cat(1, imds.Files(fold ~= i)), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    [imdsTrain,imdsValidation] = splitEachLabel(imdsTrain,0.9,0.1,'randomized');
%     [imdsTrain,imdsValidation,imdsTest] = splitEachLabel(imds,0.7,0.15,0.15,'randomized');
     net = inceptionresnetv2;lgraph = layerGraph(net);
    % figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    % plot(lgraph)
    inputSize = net.Layers(1).InputSize
    lgraph = removeLayers(lgraph, {'predictions','predictions_softmax','ClassificationLayer_predictions'});

    numClasses = numel(categories(imdsTrain.Labels));
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax')
        classificationLayer('Name','classoutput')];
    lgraph = addLayers(lgraph,newLayers);

    lgraph = connectLayers(lgraph,'avg_pool','fc');
    layers = lgraph.Layers;
    connections = lgraph.Connections;

    layers(1:820) = freezeWeights(layers(1:820));
    lgraph = createLgraphUsingConnections(layers,connections);

    pixelRange = [-30 30];
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandYReflection',true, ...
        'RandRotation',[0 360],...
        'RandScale',[0.8 1.2], ...
        'RandXShear', [0 45], ...
        'RandXShear', [0 45],...
        'RandXTranslation',pixelRange, ...
        'RandYTranslation',pixelRange);
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',imageAugmenter);
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

    options = trainingOptions('sgdm', ...
        'MiniBatchSize',12, ...
        'MaxEpochs',20, ...
        'InitialLearnRate',3e-4, ...%     'Shuffle','every-epoch', ...
        'ValidationData',augimdsValidation, ...
        'LearnRateSchedule','piecewise',...
        'ValidationFrequency',100);%, ...
%         'ValidationPatience',Inf, ...%'Verbose',false, ...
%         'Plots','training-progress');
                % lgraph = layerGraph(net);
                % figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
    %             plot(lgraph)
                % netTransfer = trainNetwork(augimdsTrain,layers,options);
    net = trainNetwork(augimdsTrain,lgraph,options);
    save(strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/ENT_10fold_all_data/',folder,'net',string(i),'.mat'),'net');
    

    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
    YPred = classify(net,augimdsTest);
    YTest = imdsTest.Labels;
    test_accuracy(i) = sum(YPred == YTest)/numel(YTest)
    test_C(:,:,i)=confusionmat(YPred,YTest)
    augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);
    YPred = classify(net,augimdsValidation);
    YVal = imdsValidation.Labels;
    val_accuracy(i) = sum(YPred == YVal)/numel(YVal)
    val_C(:,:,i)=confusionmat(YPred,YVal)
    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
    YPred = classify(net,augimdsTrain);
    YTrain = imdsTrain.Labels;
    train_accuracy(i) = sum(YPred == YTrain)/numel(YTrain)
    train_C(:,:,i)=confusionmat(YPred,YTrain)
save(strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/ENT_10fold_all_data/',folder,'acc_conf',string(i),'.mat'),'test_accuracy','test_C','val_accuracy','val_C','train_accuracy','train_C');

    netTransfer = net;
    layer = 'avg_pool';%'conv_7b_ac';%'block17_2_ac';%'mixed_7a';%'block17_20_ac';                    %% should be change according to the inceptionresnetv2 net!!!!
    featuresTrain = activations(netTransfer,augimdsTrain,layer,'OutputAs','rows');
    featuresTest = activations(netTransfer,augimdsTest,layer,'OutputAs','rows');
    
    % construct the dataset
    set = featuresTrain;

    w=net.Layers(823,1).Weights;
    save(strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/ENT_10fold_all_data/',folder,'weights',string(i),'.mat'),'w');
    set = featuresTrain;
    lookup_train=set*w';
    train_im_list = augimdsTrain.Files;
    set_test=featuresTest;
    lookup_test=set_test*w';
    featuresValidation = activations(netTransfer,augimdsValidation,layer,'OutputAs','rows');
    set_validation=featuresValidation;
    lookup_validation=set_validation*w';
    test_im_list=augimdsTest.Files;
    validation_im_list=augimdsValidation.Files;
    all_im_List=[train_im_list;validation_im_list;test_im_list];
    all_lookup_list=[lookup_train;lookup_validation;lookup_test];
    % save('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/all_lookup.mat','all_lookup_list');
    % save('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/all_im_list.mat','all_im_List');
    all_list = all_im_List(1:248);
    all_lookup = all_lookup_list;
    all_list = all_im_List(1:248);
all_lookup = all_lookup_list;

label=zeros(124,1);
k=0;
h=0;
u=0;
for j=1:124
    idx1 = find(contains(all_list,log(j,1)));
    idx2 = find(contains(all_list,log(j,3)));
    if(~isempty(idx1))&&(~isempty(idx2))
        k=k+1;
        data(k,1:2) = all_lookup(idx1,:);
        data(k,3:4) = all_lookup(idx2,:);
        data(k,5) = all_lookup(idx1,1)/all_lookup(idx2,1);
        data(k,6) = all_lookup(idx1,2)/all_lookup(idx2,2);
        data(k,7) = all_lookup(idx1,1)+all_lookup(idx2,1);
        data(k,8) = all_lookup(idx1,2)+all_lookup(idx2,2);
        data(k,9) = all_lookup(idx1,1)-all_lookup(idx2,1);
        data(k,10) = all_lookup(idx1,2)-all_lookup(idx2,2);
        data(k,11:12) = double(log(j,9:10));
        
        %read images
%         I1 = imread(strcat(location,'/', log(j,2),'/',log(j,1)));
%         I2 = imread(strcat(location,'/', log(j,4),'/',log(j,3)));
% %         %convert to the Lab color space of each
%         I1_lab=rgb2lab(I1);
%         I2_lab=rgb2lab(I2);
%         
%         I1_lab_a = mat2gray(I1_lab(:,:,2));
%         I2_lab_a = mat2gray(I2_lab(:,:,2));
%         I1_lab_b = mat2gray(I1_lab(:,:,3));
%         I2_lab_b = mat2gray(I2_lab(:,:,3));
%         
%         %find mean, median, std, and other two statistical methods
%         mean_1a = mean(mean(I1_lab_a));
%         mean_1b = mean(mean(I1_lab_b));
%         mean_2a = mean(mean(I2_lab_a));
%         mean_2b = mean(mean(I2_lab_b));
% %         
%         std_1a = std(std(I1_lab_a));
%         std_1b = std(std(I1_lab_b));
%         std_2a = std(std(I2_lab_a));
%         std_2b = std(std(I2_lab_b));
%         
%         skewness_1a = skewness(skewness(I1_lab_a));
%         skewness_1b = skewness(skewness(I1_lab_b));
%         skewness_2a = skewness(skewness(I2_lab_a));
%         skewness_2b = skewness(skewness(I2_lab_b));
%         
%         kurtosis_1a = kurtosis(kurtosis(I1_lab_a));
%         kurtosis_1b = kurtosis(kurtosis(I1_lab_b));
%         kurtosis_2a = kurtosis(kurtosis(I2_lab_a));
%         kurtosis_2b = kurtosis(kurtosis(I2_lab_b));
%         
%         %take the histograms of a and b of each
%         I1a_hist = histcounts(I1_lab_a,'NumBins',10);
%         I2a_hist = histcounts(I2_lab_a,'NumBins',10);
%         I1b_hist = histcounts(I1_lab_b,'NumBins',10);
%         I2b_hist = histcounts(I2_lab_b,'NumBins',10);
%         
% %         add these values to data
%         data(k,13:22) = I1a_hist;
%         data(k,23:32) = I1b_hist;
%         data(k,33:42) = I2a_hist;
%         data(k,43:52) = I2b_hist;
        
%         data(k,53:56) = [mean_1a mean_1b mean_2a mean_2b];
%         data(k,57:60) = [std_1a std_1b std_2a std_2b];
%         data(k,61:64) = [skewness_1a skewness_1b skewness_2a skewness_2b];
%         data(k,65:68) = [kurtosis_1a kurtosis_1b kurtosis_2a kurtosis_2b];
        
%         data(k,53:56) = [std_1a std_1b std_2a std_2b];
%         data(k,57:60)  = [skewness_1a skewness_1b skewness_2a skewness_2b];
%         data(k,61:64) = [kurtosis_1a kurtosis_1b kurtosis_2a kurtosis_2b];
        
        if(strcmp(log(j,2),'Normal'))
            h=h+1;
            label(k,:) = 1;
        else
            u=u+1;
            label(k,:) = 0;
        end
    end
    
end
data = double(data);
data = normalize(data);
data_1= data;
% data = data(:,1:end-2);
l=length(label);
% leave one image out
    trn=ones(l,1);
    test_feat = data(half_fold==i,:);
    test_label = label(half_fold==i);
    trn(half_fold==i)=0;
    train_feat = data(trn==1,:);
    train_label=label(trn==1);
    
     Mdl = TreeBagger(5,train_feat,train_label,'OOBPrediction','On',...
    'Method','classification');
%     Mdl = fitcsvm(train_feat,train_label,'OptimizeHyperparameters','auto',...
%     'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
%     'expected-improvement-plus'))
 

%         view(Mdl.Trees{1},'Mode','graph')
    [prd_lbl,scores]= predict(Mdl,test_feat);
    correct_list(half_fold==i)=str2num(cell2mat(prd_lbl))==(test_label);
    pre_tru(half_fold==i,1) = test_label;
    pre_tru(half_fold==i,2) = str2num(cell2mat(prd_lbl));
    all_scores(half_fold==i,:)= scores;
    eachfold_accuracy(i)= sum(str2num(cell2mat(prd_lbl))==(test_label)) /length(test_label);
    
    
    [XN,YN,T,AUC,OPTROCPT] = perfcurve(test_label,scores(:,2),1);
    [XAN,YAN,TA,AUCA,OPTROCPTA] = perfcurve(test_label,scores(:,1),0);
    plt = plot(XN,YN), hold on
    plt = plot(XAN,YAN), hold on
    plt = plot(OPTROCPT(1),OPTROCPT(2),'ro')
    plt = plot(OPTROCPTA(1),OPTROCPTA(2),'bo')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    legend('Normal','Abnormal','Location','Best');
    title(strcat('ROC for Fold -',string(i)));
    saveas(plt,strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Classification_results/ENT_normal_abnormal/',folder,'fold-',string(i),'_ROC.fig'));
    saveas(plt,strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Classification_results/ENT_normal_abnormal/',folder,'fold-',string(i),'_ROC.png'));
%        for s=1:legnth(prd_lbl)
%            if (str2num(cell2mat(prd_lbl))==(test_label))
%     %        if ((prd_lbl)==(test_label))
%                 correct=correct+1
%                 correct_list(half_fold==i)=1;
%            end
%        end
%         pre_tru(i,2)=(str2num(cell2mat(prd_lbl(1))));
% %         pre_tru(i,2)=(prd_lbl);
%         pre_tru(i,1)=(test_label(1));
        i
close all
save(strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Normal_Abnormal/ENT_10fold_all_data/',folder,'feat_data_fold',string(i),'.mat'),'data');

end
acc = sum(correct_list)*100/l
conf=confusionmat(pre_tru(:,1),pre_tru(:,2));
std_dev = std(correct_list)

gen_acc = mean(eachfold_accuracy)
gen_std_dev = std(eachfold_accuracy)
save(strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Classification_results/ENT_normal_abnormal/',folder,'result.mat'),'acc','correct_list','pre_tru','conf','std_dev','all_scores','half_fold','gen_acc','gen_std_dev');
% save('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Classification_results/ENT_normal_abnormal/svm_res1.mat','acc','pre_tru','conf','std_dev','data');
[XN,YN,T,AUC,OPTROCPT] = perfcurve(pre_tru(:,1),all_scores(:,2),1);
    [XAN,YAN,TA,AUCA,OPTROCPTA] = perfcurve(pre_tru(:,1),all_scores(:,1),0);
    plt = plot(XN,YN), hold on
    plt = plot(XAN,YAN), hold on
    plt = plot(OPTROCPT(1),OPTROCPT(2),'ro')
    plt = plot(OPTROCPTA(1),OPTROCPTA(2),'bo')
    xlabel('False positive rate') 
    ylabel('True positive rate')
    legend('Normal','Abnormal','Location','Best');
    title(strcat('ROC for Fold -',string(i)));
saveas(plt,strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Classification_results/ENT_normal_abnormal/',folder,'ALL_ROC.fig'));
    saveas(plt,strcat('/isilon/datalake/cialab/scratch/cialab/scamalan/Autoscope/Pair/Classification_results/ENT_normal_abnormal/',folder,'ALL_ROC.png'));
   