% Compression - 1. Zonal Method - no sorting
%               2. Thresholding - sorting
% (removing insignificant signals)

% THRESHOLDING
% Steps - 1. Sort
%         2. Square

% FUNCTIONS
% 1. find - idx = find(x > 20);
%           x(idx) = 0;
%           idx = (x > 20);

% PLOTS
% 1. MSE vs p of Total Energy
% 2. MSE vs No. of Accumulated Samples


clc;
clear all;
close all;

% Load the audio signal
[data, fs] = audioread("file_example.wav");
x = data;
x = x(:,1);
x = x(1 : 512);

% Forward DFT
X = fft(x);
stem(abs(X));
title('FFT of Original Signal');
xlabel('Frequency Index');
ylabel('|X(k)|');

% Calculate sorted energy
E_sorted = sort(abs(X).^2, "descend");
Total_E = sum(E_sorted);

% Accumulated percentage of total energy
Accumulated_E = cumsum(100 * E_sorted / Total_E);

figure;
stem(Accumulated_E);
title('Accumulated Percentage of Total Energy');
xlabel('Number of Frequency Components');
ylabel('Percentage of Total Energy');

% Percentage of total energy to be retained
p = 90;

% Find threshold corresponding to p% of total energy
for j = 1 : length(Accumulated_E)
    if Accumulated_E(j) >= p
        threshold = E_sorted(j);
        break;
    end
end

% Remove insignificant frequency components
X_thresholded = X;

for j = 1:length(X)
    if abs(X(j))^2 < threshold
        X_thresholded(j) = 0;
    end
end

% Inverse DFT
real_X = real(ifft(X_thresholded));

% Plot approximate signal
figure;
stem(real_X);
title('Approximate Signal after Thresholding');
xlabel('Sample Number');
ylabel('Amplitude');

% Calculate Mean Square Error
MSE = mean((x - real_X).^2);

fprintf('Percentage of Total Energy Retained = %.2f%%\n', p);
fprintf('Threshold = %.6e\n', threshold);
fprintf('Mean Square Error = %.6e\n', MSE);

% MSE vs Percentage of Total Energy
p_values = 1:1:100;
MSE_p = zeros(size(p_values));

for k = 1:length(p_values)

    p = p_values(k);

    % Find threshold for current percentage
    for j = 1:length(Accumulated_E)
        if Accumulated_E(j) >= p
            threshold = E_sorted(j);
            break;
        end
    end

    % Apply threshold
    X_temp = X;

    for j = 1:length(X)
        if abs(X(j))^2 < threshold
            X_temp(j) = 0;
        end
    end

    % Reconstruct signal
    x_temp = real(ifft(X_temp));

    % Calculate MSE
    MSE_p(k) = mean((x - x_temp).^2);

end

figure;
plot(p_values, MSE_p, 'LineWidth', 2);
xlabel('Percentage of Total Energy Retained (%)');
ylabel('Mean Square Error');
title('MSE vs Percentage of Total Energy');
grid on;

% MSE vs Number of Accumulated Samples
MSE_samples = zeros(length(E_sorted), 1);

for k = 1 : length(E_sorted)

    threshold = E_sorted(k);
    X_temp = X;

    for j = 1:length(X)
        if abs(X(j))^2 < threshold
            X_temp(j) = 0;
        end
    end

    x_temp = real(ifft(X_temp));
    MSE_samples(k) = mean((x - x_temp).^2);

end

figure;
plot(1:length(E_sorted), MSE_samples, 'LineWidth', 2);
xlabel('Number of Accumulated Samples');
ylabel('Mean Square Error');
title('MSE vs Number of Accumulated Samples');
grid on;
