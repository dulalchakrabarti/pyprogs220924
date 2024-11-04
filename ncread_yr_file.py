import imdlib as imd
start_yr = 1901
end_yr = 1909
var_type = 'rain'
file_dir='./rain'
data = imd.open_data(var_type, start_yr, end_yr, 'yearwise',file_dir)
ds = data.get_xarray()
df = ds.to_dataframe()
print(df.head())

