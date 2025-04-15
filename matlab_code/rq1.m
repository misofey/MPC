t = linspace(0, 1, 1000);
vin = 2.5*sin(100*t+23/180*pi);
tf(lowpass(vin, 15/100)