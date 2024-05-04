figure;

[f,Fs] = audioread('lick_samplet\36_lick.wav');
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