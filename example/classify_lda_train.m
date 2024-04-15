function model = classify_lda_train(F, Fk, varargin)
% Use:
%   model = classify_lda_train(F, Fk [, varargin])
% 
% Input:
%   F                       Feature vector, [observations X features]
%   Fk                      Class labels, [observations X 1]. Only two
%                           classes allowed
% Optional:
%   'priors', [val1 val2]   prior probabilities for the two classes
%   'estimation', 'type'    Covariance estimation:
%                           - 'estimation', 'none'                  
%                               No estimation [DEFAULT]
%                           - 'estimation', 'shrink', 'lambda', val
%                               Shrinkage of cavariance matrices with lambda
%                               parameter
% 
% Output:
%   model.type              Type of discriminant analysis (lda) 
%        .m1                Mean of the first class
%        .m2                Mean of the second class
%        .cov               Covariance
%        .priors            Prior probabilities for the two classes
%        .estimation        Estimation type ('none', 'shrink')
%        .lambda            Lambda parameter for shrinkage
% 
% See also classify_lda_eval, classify_lda_example, classify_train,
% classify_eval


 %% Handling input arguments
    if nargin < 2
        error('[lda] - Error. Not enogh input arguments');
    end
    
    def_priors      = [0.5 0.5];
    def_estimation  = 'none';
    def_lambda      = nan;
    
    pnames  = {'priors',  'estimation',   'lambda'};
    default = {def_priors, def_estimation, def_lambda};
    
%     [~, msg, priors, estimation, lambda] = util_getargs(pnames, default, varargin{:});
    msg = '';
    priors =  [0.5 0.5];
    estimation = 'shrink';
    lambda = 0.5;

    
    if isempty(msg) == false
        error(['[lda] - ' msg]);
    end
    
    if isequal(sum(priors), 1) == false
        error('[lda] - The provided priors do not sum to 1');
    end

    %% Dataset creations
    classes = unique(Fk);
    if isequal(length(classes), 2) == false
        error('[lda] - Number of classes must be equal to 2');
    end

   % Classes mean computation
   m1 = mean(F(Fk == classes(1),:))';
   m2 = mean(F(Fk == classes(2),:))';
   
   % Covariance computation
   covariance = cov(F);
    
    % Estimation of the covariance
    switch estimation
        case 'none'         
        case 'shrink'
            if size(F, 2) == 1
                warning('[lda] - Skipping regularization. Cannot shrink covariance with only one feature');
                estimation = 'none';
                lambda     = nan;
            else
                covariance  = (1 - lambda)*covariance + (lambda/size(F, 2))*trace(covariance)*eye(size(covariance));
            end
        otherwise
            error('[lda] - Unknown covariance type');
    end
    
    
    model.type       = 'lda';
    model.m1         = m1;
    model.m2         = m2;
    model.cov        = covariance;
    model.classes    = classes;
    model.priors     = priors;
    model.estimation = estimation;
    model.lambda     = lambda;

end
