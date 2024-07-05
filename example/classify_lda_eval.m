function [pp, class] = classify_lda_eval(model, F)
% Use:
%   [pp, class] = classify_lda_eval(model,F)
% 
% Input [required]:
%   model.type              Type of discriminant analysis (lda) 
%        .m1                Mean of the first class
%        .m2                Mean of the second class
%        .cov               Covariance
%        .priors            Prior probabilities for the two classes
%     
%   F                       Feature vector [observations X features]
%
%  Output:
%   pp                      Posterior probabilities for the two classes
%   class                   Predicted labels
% 
% See also classify_lda_train, classify_lda_example, classify_train,
% classify_eval

if strcmpi(model.type, 'lda') == 0
    error('[lda] - The provided model is not a lda');
end

% Parameters extraction from the input model
priors     = model.priors;
m1         = model.m1;
m2         = model.m2;
covariance = model.cov;
classes    = model.classes;

% Output parameters initialization
NumObservations = size(F, 1);
pp              = zeros(NumObservations, 2);
class           = zeros(NumObservations, 1);

for oId = 1:NumObservations
    
    % Single observation
    x = F(oId, :)';
    
    % Class belonging probabilities
    lh(1) = (1/(sqrt(((2*pi)^length(x))*det(covariance))))*exp(-0.5*(x-m1)'/(covariance)*(x-m1));
    lh(2) = (1/(sqrt(((2*pi)^length(x))*det(covariance))))*exp(-0.5*(x-m2)'/(covariance)*(x-m2));
%     lh(1) = mvnpdf(x, m1, covariance);
%     lh(2) = mvnpdf(x, m2, covariance);
    
    % Posterior class belonging probabilities
    post(1) = lh(1)*priors(1)/(priors(1)*lh(1)+priors(2)*lh(2));
    post(2) = lh(2)*priors(2)/(priors(1)*lh(1)+priors(2)*lh(2));
    
    % Class prediction
    pp(oId,:) = post;
    [~, idcls] = max(post);
    class(oId) = classes(idcls);
end