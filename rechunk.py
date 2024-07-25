import sys
import intake
import xarray as xr
from matplotlib import pyplot as plt
import glob
import warnings
from dask.distributed import Client
from rechunker import rechunk

def setup_dask_client():
    client = Client()
    return client

def load_catalogue(catalogue_path):
    mRuns = sorted(glob.glob(catalogue_path+'/*ssp370*.json'))
    return mRuns

def get_model_name(file):
    mtmp = file.split('/')
    model_name = mtmp[-1].split('.')
    return model_name[0]

def load_variables(cat):
    variables = ['pr', 'tasmax', 'hursmax', 'hursmin', 'sfcWindmax', 'tasmin']
    data = {}
    for var in variables:
        data[var] = cat.search(variable={var}).to_dask(cdf_kwargs={'chunks': {'time': 1, 'lat': 691, 'lon': 886}})
    return data

def reindex_variables(data, common_time):
    for var in data:
        data[var] = data[var].reindex(time=common_time)
    return data

def remove_height_coords(data, ACS_model):
    if ACS_model == 'BOM':
        coords_to_drop = ['height', 'level_height', 'model_level_number', 'sigma']
        for var in data:
            for coord in coords_to_drop:
                if coord in data[var].coords:
                    data[var] = data[var].reset_coords(coord, drop=True)
    return data

def merge_datasets(data):
    return xr.merge([data[var] for var in data], compat='override')

def rechunk_data(ds, target_store, temp_store, target_chunks, max_mem):
    array_plan = rechunk(ds, target_chunks, max_mem, target_store, temp_store=temp_store)
    array_plan.execute()

def process_run(run_index, file, target_store_base):
    print(f"Running with index: {run_index}")
    cat = intake.open_esm_datastore(file)
    model_name = get_model_name(file)

    data = load_variables(cat)
    common_time = data['pr'].coords['time']
    data = reindex_variables(data, common_time)

    ACS_model = model_name.split('_')[1]
    data = remove_height_coords(data, ACS_model)

    ds = merge_datasets(data)

    target_chunks = {"time": len(data['pr'].time), "lat": 33, "lon": 43}
    max_mem = "80GB"
    target_store = f"{target_store_base}/{model_name}.zarr"
    temp_store = f"{target_store_base}/tmp.zarr"

    # Ensure the target directories are clean
    !rm -rf {target_store}
    !rm -rf {temp_store}

    rechunk_data(ds, target_store, temp_store, target_chunks, max_mem)
    print(f'run {run_index} complete')

def main():
    warnings.filterwarnings('ignore')

    catalogue_path = '/g/data/ia39/catalogues'
    target_store_base = "/scratch/xv83/ep5799"

    client = setup_dask_client()
    mRuns = load_catalogue(catalogue_path)

    file = mRuns[12]
    process_run(12, file, target_store_base)


if __name__ == "__main__":
    main()
