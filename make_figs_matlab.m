function make_figs_matlab()
% Make polished figures from pipeline outputs.
% Outputs: outputs/figs/fig1_*.png, fig2_*.png, fig3_*.png

%% ==== Paths & params ====
proj = '/Users/lynn/Documents/351/project';
slicesCSV  = fullfile(proj,'data/meta/slices.csv');
indexCSV   = fullfile(proj,'data/meta/features_index.csv');
outdir     = fullfile(proj,'outputs/figs'); if ~exist(outdir,'dir'), mkdir(outdir); end

classes = {'siren','car_horn','engine','construction','other'}; % 想展示的类别
examplesPerClass = 1;   % 每类展示几张谱图
maxPerClassMean  = 300; % 计算均值谱时每类最多取多少个切片
sr = 16000; nMels = 64; fmin = 50; fmax = 8000; win_ms = 25; hop_ms = 10;

% 统一美观样式
set(groot,'DefaultAxesFontName','Helvetica',...
          'DefaultAxesFontSize',11,...
          'DefaultAxesLineWidth',1.0,...
          'DefaultLineLineWidth',1.8,...
          'DefaultFigureColor','w');
co = lines(8); set(groot,'DefaultAxesColorOrder',co);

%% ==== Read tables ====
Tidx = readtable(indexCSV,'TextType','string');
if ~ismember("label", Tidx.Properties.VariableNames)
    error("features_index.csv must contain a 'label' column.");
end
Tidx.label = fillmissing(Tidx.label,'constant',"other");

Tslices = readtable(slicesCSV,'TextType','string'); % 用于从 slice_path 取音频
if ~ismember("slice_path", Tslices.Properties.VariableNames)
    error("slices.csv must contain 'slice_path' and 'label' columns.");
end

%% ==== Figure 1: Label counts ====
counts = groupsummary(Tidx,"label");
[~,ord] = sort(counts.GroupCount,'descend');
counts = counts(ord,:);

fh1 = figure('Position',[100 100 720 380]);
bar(categorical(counts.label), counts.GroupCount, 'FaceColor', [0.2 0.45 0.85]);
ylabel('Slices'); title('Label distribution (slices)');
grid on; box off; set(gca,'XTickLabelRotation',30);
text(1:numel(counts.label), counts.GroupCount, string(counts.GroupCount),...
    'VerticalAlignment','bottom','HorizontalAlignment','center','FontSize',9);
exportgraphics(fh1, fullfile(outdir,'fig1_label_counts.png'), 'Resolution',300);

%% ==== Figure 2: Log-Mel examples (grid) ====
% 选出每类样例（从 slices.csv 找 wav）
sel = table(); 
for i = 1:numel(classes)
    c = classes{i};
    rows = Tslices(Tslices.label==c, :);
    if height(rows)>0
        k = min(examplesPerClass, height(rows));
        sel = [sel; rows(1:k,:)];
    end
end

nrows = numel(classes); ncols = examplesPerClass;
fh2 = figure('Position',[100 100 900 260+220*(nrows-1)]);
tlo = tiledlayout(nrows, ncols, 'TileSpacing','compact','Padding','compact');

for i = 1:height(sel)
    nexttile;
    try
        [x,fs] = audioread(sel.slice_path(i));
        if fs~=sr, x = resample(x, sr, fs); fs = sr; end
        Sdb = computeLogMel(x, fs, nMels, fmin, fmax, win_ms, hop_ms); % dB, [nMels x T]
        imagesc(Sdb); axis xy;

        % 统一 colormap/clim
        try, colormap(turbo); catch, colormap(parula); end
        caxis([-80 0]);
        xlabel('Frames'); ylabel('Mel bins');
        title(sprintf('%s', sel.label(i)), 'FontWeight','bold');
    catch ME
        text(0.5,0.5, "read fail", 'HorizontalAlignment','center'); axis off
        warning("%s", ME.message);
    end
end
cb = colorbar; cb.Layout.Tile = 'east'; cb.Label.String = 'dB';
title(tlo,'Representative log-Mel spectrograms');
exportgraphics(fh2, fullfile(outdir,'fig2_logmel_examples.png'), 'Resolution',300);

%% ==== Figure 3: Mean Mel profiles ±1σ ====
fh3 = figure('Position',[100 100 760 380]); hold on; grid on; box off
legendEntries = {};
ax = gca;
colorOrder = get(ax,'ColorOrder');   % ✅ 先取出颜色表再索引
nColors = size(colorOrder,1);

for i = 1:numel(classes)
    c = classes{i};
    rows = Tslices(Tslices.label==c, :);
    if isempty(rows), continue; end
    if height(rows) > maxPerClassMean
        rows = rows(randperm(height(rows), maxPerClassMean), :);
    end
    vecs = [];
    for j = 1:height(rows)
        try
            [x,fs] = audioread(rows.slice_path(j));
            if fs~=sr, x = resample(x, sr, fs); fs = sr; end
            Sdb = computeLogMel(x, fs, nMels, fmin, fmax, win_ms, hop_ms); % [nMels x T]
            v = mean(Sdb,2,'omitnan'); % [nMels x 1]
            vecs(:,end+1) = v; %#ok<AGROW>
        catch
        end
    end
    if isempty(vecs), continue; end
    mu = mean(vecs,2,'omitnan'); sd = std(vecs,0,2,'omitnan');
    f = melCenters(nMels, fmin, fmax); % 频率轴（Hz）

    clr = colorOrder(mod(i-1, nColors)+1, :);  % ✅ 合法索引
    % 先画阴影，再画均值线，保证线在上方
    fill([f, fliplr(f)], [ (mu - sd).', fliplr((mu + sd).') ], ...
         clr, 'FaceAlpha',0.15, 'EdgeColor','none');
    plot(f, mu, 'Color', clr, 'LineWidth',2);

    legendEntries{end+1} = sprintf('%s (n=%d)', c, size(vecs,2)); %#ok<AGROW>
end
xlabel('Frequency (Hz, mel centers)'); ylabel('log-Mel (dB)');
title('Mean mel profiles with \pm1 std');
legend(legendEntries,'Location','northeast');
exportgraphics(fh3, fullfile(outdir,'fig3_mean_mel_profiles.png'), 'Resolution',300);

disp("[OK] Saved figures to " + outdir);
end

%% ===== helpers =====
function Sdb = computeLogMel(x, fs, nMels, fmin, fmax, win_ms, hop_ms)
win = round(win_ms/1000*fs); hop = round(hop_ms/1000*fs);
nfft = 2^nextpow2(max(win, 256));
x = mean(x,2); x = x(:);
if exist('melSpectrogram','file')==2
    [S,~,~] = melSpectrogram(x, fs, ...
        'Window',hann(win,'periodic'), 'OverlapLength',win-hop, ...
        'FFTLength', nfft, 'NumBands', nMels, ...
        'FrequencyRange',[fmin fmax], 'SpectrumType','power');
    Sdb = pow2db(S + 1e-10);
else
    % fallback: 普通功率谱 + 自建 mel 滤波器组
    [S,F,~] = spectrogram(x, hann(win,'periodic'), win-hop, nfft, fs, 'yaxis');
    P = abs(S).^2; % power
    H = designMelBank(F, nMels, fmin, fmax); % [nMels x numFreq]
    M = H * P; Sdb = pow2db(M + 1e-10);
end
end

function H = designMelBank(F, nMels, fmin, fmax)
% 简易三角 mel 滤波器组合成
mel = @(f) 2595*log10(1+f/700);
melInv = @(m) 700*(10.^(m/2595)-1);
m = linspace(mel(fmin), mel(fmax), nMels+2);
fc = melInv(m); % nMels+2
H = zeros(nMels, numel(F));
for i=1:nMels
    f1=fc(i); f2=fc(i+1); f3=fc(i+2);
    up = (F>=f1 & F<=f2) .* ((F-f1)/(f2-f1));
    dn = (F>=f2 & F<=f3) .* ((f3-F)/(f3-f2));
    H(i,:) = up + dn;
end
end

function f = melCenters(nMels, fmin, fmax)
mel = @(f) 2595*log10(1+f/700);
melInv = @(m) 700*(10.^(m/2595)-1);
m = linspace(mel(fmin), mel(fmax), nMels+2);
f = melInv(m(2:end-1));
end