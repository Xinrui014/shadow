%% compute RMSE(MAE)|计算RMSE(MAE) 
clear;close all;clc
% 1`modify the following directories 2`run|修改路径,再运行

% GT mask directory|掩膜路径
maskdir = 'C:\projects\ShadowDiffusion\dataset\ISTD_adjusted_C\test\test_B\';
MD = dir([maskdir '\*.png']);

% result directory|结果路径
shadowdir = 'C:\projects\HomoFormer\log\ostu_mask_ISTD+\metrics_before_process\';  %test_A
SD = dir([shadowdir '\*_sr.png']);

    
% ground truth directory|GT路径
freedir =shadowdir; %AISTD
FD = dir([freedir '\*_hr.png']);

total_dists = 0;
total_pixels = 0;
total_distn = 0;
total_pixeln = 0;
allmae=zeros(1,size(SD,1)); 
smae=zeros(1,size(SD,1)); 
nmae=zeros(1,size(SD,1)); 
ppsnr=zeros(1,size(SD,1));
ppsnrs=zeros(1,size(SD,1));
ppsnrn=zeros(1,size(SD,1));
sssim=zeros(1,size(SD,1));
sssims=zeros(1,size(SD,1));
sssimn=zeros(1,size(SD,1));
rrmse=zeros(1,size(SD,1));
rrmses=zeros(1,size(SD,1));
rrmsen=zeros(1,size(SD,1));
cform = makecform('srgb2lab');

for i=1:size(SD)
    %disp(SD(i));
    %disp(FD(i));
    %disp(MD(i));
    sname = strcat(shadowdir,SD(i).name); 
    fname = strcat(freedir,FD(i).name); 
    mname = strcat(maskdir,MD(i).name); 
    s=imread(sname);
    f=imread(fname);
    m=imread(mname);
    
    f = double(f)/255;
    s = double(s)/255;
    
     s=imresize(s,[256 256]);
     f=imresize(f,[256 256]);
     m=imresize(m,[256 256]);


    nmask=~m;       %mask of non-shadow region|非阴影区域的mask
    smask=~nmask;   %mask of shadow regions|阴影区域的mask
    
    ppsnr(i)=psnr(s,f);
    ppsnrs(i)=psnr(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    ppsnrn(i)=psnr(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));
    sssim(i)=ssim(s,f);
    sssims(i)=ssim(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
    sssimn(i)=ssim(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));

   
    %% RMSE
%     arrmse=rmse(s,f);
%     rrmses(i)=rmse(s.*repmat(smask,[1 1 3]),f.*repmat(smask,[1 1 3]));
%     rrmsen(i)=rmse(s.*repmat(nmask,[1 1 3]),f.*repmat(nmask,[1 1 3]));


    f = applycform(f,cform);    
    s = applycform(s,cform);
    
    %% MAE, per image
    dist=abs((f - s));
    sdist=dist.*repmat(smask,[1 1 3]);
    sumsdist=sum(sdist(:));
    ndist=dist.*repmat(nmask,[1 1 3]);
    sumndist=sum(ndist(:));
    
    sumsmask=sum(smask(:));
    sumnmask=sum(nmask(:));
    
    %% MAE, per pixel
    allmae(i)=sum(dist(:))/size(f,1)/size(f,2);
    smae(i)=sumsdist/sumsmask;
    nmae(i)=sumndist/sumnmask;
    
    total_dists = total_dists + sumsdist;
    total_pixels = total_pixels + sumsmask;
    
    total_distn = total_distn + sumndist;
    total_pixeln = total_pixeln + sumnmask;  


    disp(i);
end
fprintf('PSNR(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(ppsnr),mean(ppsnrn),mean(ppsnrs));
fprintf('SSIM(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(sssim),mean(sssimn),mean(sssims));
% fprintf('RMSE(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(rrmse),mean(rrmsen),mean(rrmses));
fprintf('PI-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n',mean(allmae),mean(nmae),mean(smae));
fprintf('PP-Lab(all,non-shadow,shadow):\n%f\t%f\t%f\n\n',mean(allmae),total_distn/total_pixeln,total_dists/total_pixels);
