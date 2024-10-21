%% Penumbra Mask Creation and PSNR Calculation
clear; close all; clc

% Directories
maskdir = 'C:\projects\ShadowDiffusion\experiments_lightning\joint_tune_sam_diffusion\mse+1e-1contGradnoPenumbra\399_old_2\';
%maskdir = 'C:\projects\ShadowDiffusion\dataset\SRD_DHAN_mask_old\test\test_B\'
shadowdir = 'C:\projects\ShadowDiffusion\experiments_lightning\joint_tune_sam_diffusion\mse+1e-1contGradnoPenumbra\399_old_2\';
%freedir = 'C:\projects\ShadowDiffusion\experiments_lightning\joint_tune_sam_diffusion\mse+1e-1contGradnoPenumbra\399_old_2\';
%freedir = 'C:\projects\ShadowDiffusion\experiments_lightning\inference\orig_ShadowDiffusion\results\'
freedir = 'C:\papers\shadow_removal\CompareResults\SRD\Inpaint4shadow\De-shadowed_results_SRD\'
penumbradir = 'C:\papers\shadow_removal\CompareResults\SRD\ours\penumbra_mask_DHAN';

% Create penumbra mask output directory if it doesn't exist
if ~exist(penumbradir, 'dir')
    mkdir(penumbradir);
end

% Get file lists
MD = dir([maskdir '*_soft_mask.png']);
SD = dir([shadowdir '*_sr.png']);
FD = dir([freedir '*.png']);

% Initialize arrays for metrics
ppsnr = zeros(1, length(SD));
sssim = zeros(1, length(SD));
allmae = zeros(1, length(SD));

% Parameters
threshold = 100;
dilation_size = 7;
eroded_size = 7;

% Create structuring element for dilation/erosion
de = strel('disk', dilation_size);
se = strel('disk', eroded_size);

for i = 1:length(SD)
    % Read images
    mname = fullfile(maskdir, MD(i).name);
    sname = fullfile(shadowdir, SD(i).name);
    fname = fullfile(freedir, FD(i).name);
    
    m = imread(mname);
    s = im2double(imread(sname));
    f = im2double(imread(fname));

    [h, w, ~] = size(s);
    f = imresize(f, [h, w]);
    m = imresize(m, [h, w]);
    
    % Create binary mask
    binary_mask = imbinarize(m, threshold/255);
    
    % Create penumbra mask
    dilated = imdilate(binary_mask, de);
    eroded = imerode(binary_mask, se);
    penumbra_mask = dilated - eroded;
    
    % Save penumbra mask
    penumbra_name = fullfile(penumbradir, ['penumbra_' MD(i).name]);
    imwrite(penumbra_mask, penumbra_name);
    
    % Resize images if necessary
    [h, w, ~] = size(s);
    f = imresize(f, [h, w]);
    penumbra_mask = imresize(penumbra_mask, [h, w]);
    
    % Calculate metrics in penumbra area
    penumbra_mask_3d = repmat(penumbra_mask, [1, 1, 3]);
    s_penumbra = s .* penumbra_mask_3d;
    f_penumbra = f .* penumbra_mask_3d;
    
    ppsnr(i) = psnr(s_penumbra, f_penumbra);
    sssim(i) = ssim(s_penumbra, f_penumbra);
    
    % Calculate MAE in LAB color space
    cform = makecform('srgb2lab');
    s_lab = applycform(s, cform);
    f_lab = applycform(f, cform);
    dist = abs(f_lab - s_lab);
    penumbra_dist = dist .* penumbra_mask_3d;
    allmae(i) = sum(penumbra_dist(:)) / sum(penumbra_mask(:));
    
    disp(['Processed image ' num2str(i) ' of ' num2str(length(SD))]);
end

% Display results
fprintf('PSNR (penumbra area): %f\n', mean(ppsnr));
fprintf('SSIM (penumbra area): %f\n', mean(sssim));
fprintf('MAE-Lab (penumbra area): %f\n', mean(allmae));