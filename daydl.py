import imdlib as imd
start_dy = '2024-11-01'
end_dy = '2024-11-01'
var_type = 'rain'
file_dir='./rain'
data = imd.open_real_data(var_type, start_dy, end_dy, file_dir)
