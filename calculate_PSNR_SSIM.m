clear all; close all;clc;
% Parse command line arguments
path = "./experiments_lightning/test_with_LPfilter/results_kernel_7";

% Get file names
real_names = dir(fullfile(path, '*_hr.png'));
fake_names = dir(fullfile(path, '*_sr.png'));

% Sort file names
real_names = sort({real_names.name});
fake_names = sort({fake_names.name});

avg_psnr = 0;
avg_ssim = 0;
idx = 0;

for i = 1:length(real_names)
    rname = fullfile(path, real_names{i});
    fname = fullfile(path, fake_names{i});

    idx = idx + 1;
    ridx = strsplit(real_names{i}, '_hr');
    fidx = strsplit(fake_names{i}, '_sr');
    assert(strcmp(ridx{1}, fidx{1}), sprintf('Image ridx:%s!=fidx:%s', ridx{1}, fidx{1}));

    hr_img = imread(rname);
    sr_img = imread(fname);
    hr_img = double(hr_img)/255;
    sr_img = double(sr_img)/255;
    psnr_eval = psnr(sr_img, hr_img);
    ssim_val = ssim(sr_img, hr_img);
    avg_psnr = avg_psnr + psnr_eval;
    avg_ssim = avg_ssim + ssim_val;

    if mod(idx, 20) == 0
        fprintf('Image:%d, PSNR:%.4f, SSIM:%.4f\n', idx, psnr_eval, ssim_val);
    end
end

avg_psnr = avg_psnr / idx;
avg_ssim = avg_ssim / idx;

% Log
fprintf('# Validation # PSNR: %.4e\n', avg_psnr);
fprintf('# Validation # SSIM: %.4e\n', avg_ssim);