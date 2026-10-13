[data, fs] = audioread("file_example.wav");  % File Path
x = data;
x = x(:,1);
x = x(1 : 512);
N = length(x);

figure;
plot(1 : N, x);
xlabel('Sample No.');
ylabel('Amplitude');
title('Original Audio Signal');
grid on;

X = fft(x);

figure;
plot(0 : N - 1, abs(X));
xlabel('Frequency Bin');
ylabel('X|k|');
title('FFT Magnitude Spectrum');
grid on;

zone_size = round(N * 0.2);
window = zeros(N, 1);

X_shifted = fftshift(X);
center = floor(N / 2) + 1;
half_width = floor(zone_size / 2);

start_ind = center - half_width;
end_ind = center + half_width - 1;

start_ind = max(start_ind, 1);
end_ind = min(end_ind, N);

window(start_ind : end_ind) = 1;

X_approx_shifted = X_shifted .* window;
X_approx = ifftshift(X_approx_shifted);
x_approx = real(ifft(X_approx));

mse_20 = mean((x - x_approx) .^ 2);

figure;
plot(1 : N, x, 'LineWidth', 1);
hold on;
plot(1 : N, x_approx, 'LineWidth', 1);
xlabel('Sample No.');
ylabel('Amplitude');
title('Original Signal vs Approximate Signal (20% Zone)');
legend('Original Signal', 'Approximate Signal');
grid on;

figure;
plot(0 : N - 1, abs(X_shifted));
hold on;
plot(0 : N - 1, abs(X_approx_shifted));
xlabel('Frequency Bin');
ylabel('Magintude');
title('Original FFT vs 20% Zonal FFT');
legend('Original FFT', 'Zonal FFT');
grid on;

percentage_values = 0 : 1 : 100;
mse_values = zeros(size(percentage_values));

for p = 1 : length(percentage_values)
    percentage = percentage_values(p);
    zone_size = round((percentage / 100) * N);
    window = zeros(N, 1);
    
    if zone_size > 0
        half_width = floor(zone_size / 2);
        center = floor(N / 2) + 1;

        start_ind = center - half_width;
        end_ind = center + half_width - 1;

        start_ind = max(start_ind, 1);
        end_ind = min(end_ind, N);

        window(start_ind : end_ind) = 1;
    end

    X_approx_shifted = X_shifted .* window;
    X_approx = ifftshift(X_approx_shifted);
    x_approx = real(ifft(X_approx));

    mse_values(p) = mean((x - x_approx).^2);

end

figure;
plot(percentage_values, mse_values, 'LineWidth', 2);
xlabel('Zone Size (%)');
ylabel("Mean Square Error (MSE)");
title('Zone Distortion Curve');
grid on;
