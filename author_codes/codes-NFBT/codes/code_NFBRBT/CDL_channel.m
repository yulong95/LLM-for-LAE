function h = CDL_channel(x,f_c,Lambda)

N = length(x);

cdl_model = nrCDLChannel('ChannelFiltering', false);
cdl_model.MaximumDopplerShift = 50;
cdl_model.SampleRate = 10e2;
cdl_model.SampleDensity = Inf;
cdl_model.NumTimeSamples = 1;
cdl_model.TransmitAntennaArray.Size = [N, 1, 1, 1, 1]; % [M N P Mg Ng]
cdl_model.ReceiveAntennaArray.Size = [1, 1, 1, 1, 1]; 
cdl_model.Seed = randi(10000);
cdl_model.TransmitAntennaArray.ElementSpacing = [(x(2)-x(1))/Lambda, (x(2)-x(1))/Lambda, 0, 0];
cdl_model.CarrierFrequency = f_c;

[pathGains, ~] = cdl_model();
h = reshape(sum(pathGains(1, :, :, :), 2), [N, 1]);

% h = h/norm(h);
end

