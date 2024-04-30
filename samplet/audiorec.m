figure;

[f,Fs] = audioread('Tallenne.m4a');
N = length(f);
slength = N/Fs;
tiledlayout(2,1)

nexttile
t = linspace(0, N/Fs, N);
plot(t, f);
title('signal')

nexttile
f = f(1:Fs);
spectrogram(f, 256, [], [], Fs, 'yaxis');
title('Spectrogram')